function fgInitCamera(root) {
  /* 一个页面可能有多个播放器（护士长「病区画面」最多 9 格）。
      从「只绑第一个 [data-camera-root]」改成逐根初始化：
     单画面页仍只有一个根，行为不变；宫格建好格子后调 window.fgInitCamera(el)。
     重复调用直接返回 —— 重建格子时不会绑两遍点击。 */
  if (!root || root.__fgInited) return;
  root.__fgInited = true;

  var video = root.querySelector('[data-camera-video]');
  var ph = root.querySelector('[data-camera-ph]');
  var badge = root.querySelector('[data-camera-badge]');
  var hint = root.querySelector('[data-camera-hint]');
  var stateEl = root.querySelector('[data-camera-state]');
  var btn = root.querySelector('[data-camera-toggle]');

  var STREAM = (root.getAttribute('data-camera-stream') || '').trim();
  /*- ：护士长「病区画面」只要状态探测、不要自动建流（她可能整天开着，
      2.5 Mbps 一路就吃掉出网带宽的一半）。根节点写 data-camera-autoplay="0" 即可关闭；
      不带这个属性 = 原行为（护士站点位画面页进页面即播）。 -*/
  var AUTOPLAY = root.getAttribute('data-camera-autoplay') !== '0';

  var HLS_JS = '/static/vendor/hls.light.min.js';
  var WHEP_URL = '/api/v1/live/whep/' + encodeURIComponent(STREAM);
  var HLS_PLAYLIST = '/api/v1/live/hls/' + encodeURIComponent(STREAM) + '/index.m3u8';
  var STATUS_URL = '/api/v1/live/status/' + encodeURIComponent(STREAM);
  /* 冷却按流隔离：一个演示点位失败，不该拖累真摄像头 */
  var COOLDOWN_KEY = 'fg_cam_whep_fail:' + STREAM;
  var WHEP_COOLDOWN_MS = 10 * 60 * 1000;

  var pc = null;
  var sessionUrl = null;
  var hls = null;
  var busy = false;
  var playing = false;
  var gen = 0;
  var autoStarted = false;

  /* ── 界面状态 ─────────────────────────────────────────────── */
  function setBanner(title, desc) {
    if (badge) badge.textContent = title;
    if (hint) hint.textContent = desc || '';
    if (ph) ph.hidden = false;
    if (video) video.hidden = true;
    root.classList.remove('is-playing');
  }

  function setDot(text, cls) {
    if (!stateEl) return;
    stateEl.textContent = text;
    stateEl.className = 'fg-camera__state' + (cls ? ' ' + cls : '');
  }

  function setBtn(text, disabled) {
    if (!btn) return;
    btn.textContent = text;
    btn.disabled = !!disabled;
    btn.setAttribute('aria-pressed', text === '停止' ? 'true' : 'false');
  }

  function showVideo() {
    playing = true;
    if (video) {
      video.hidden = false;
      var p = video.play();
      if (p && p.catch) p.catch(function () { /* 自动播放被拦时用原生控件 */ });
    }
    if (ph) ph.hidden = true;
    root.classList.add('is-playing');
    setDot('播放中', 'is-live');
    setBtn('停止', false);
  }

  /* 用户主动停 / 切后台停：把界面也退回"未播放"，不能只停媒体还写着"播放中" */
  function resetUi(title, desc) {
    setBanner(title || '已停止', desc || '');
    setDot('已停止', 'is-off');
    setBtn('播放', !STREAM);
  }

  /* ── 冷却 ─────────────────────────────────────────────────── */
  function webrtcRecentlyFailed() {
    try {
      var t = parseInt(sessionStorage.getItem(COOLDOWN_KEY) || '0', 10);
      return t > 0 && (Date.now() - t) < WHEP_COOLDOWN_MS;
    } catch (e) { return false; }
  }

  function markWebrtcFailed() {
    try { sessionStorage.setItem(COOLDOWN_KEY, String(Date.now())); } catch (e) {}
  }

  /* ── 收尾 ─────────────────────────────────────────────────── */
  /* 只收媒体与上游会话，不动 gen/界面。ICE 断了的回落路径要用它：
     那里必须保住本次起播的身份，否则 fallback 自己就被 stop() 作废了。 */
  function teardownMedia() {
    /* 骨架叠加层靠这个全局拿 RTCPeerConnection 去读 jitterBufferDelay 估视频延迟。
       不置空的话，停播后它还会对一条已经 close 的 pc 调 getStats。 */
    root.__fgLivePc = null;
    if (hls) {
      try { hls.destroy(); } catch (e) {}
      hls = null;
    }
    if (pc) {
      try { pc.close(); } catch (e) {}
      pc = null;
    }
    if (sessionUrl) {
      /* 必须 DELETE，否则 mediamtx 一直按"有人在看"往外送流 */
      try { fetch(sessionUrl, { method: 'DELETE', keepalive: true }); } catch (e) {}
      sessionUrl = null;
    }
    if (video) {
      try { video.srcObject = null; } catch (e) {}
      video.removeAttribute('src');
      try { video.load(); } catch (e) {}
    }
  }

  function stop() {
    gen++;              /* 作废所有在途的起播/回落链 */
    busy = false;
    playing = false;
    teardownMedia();
    if (video) video.hidden = true;
    root.classList.remove('is-playing');
  }

  /* ── WebRTC ───────────────────────────────────────────────── */
  /* alive(): 本次起播是否仍然有效（stop() 之后为假）
     onLost(): 已经连上、之后 ICE 才断——不能再 reject（Promise 已 settle），
               交给它去回落 HLS，否则页面会永远停在"播放中"的黑屏上。 */
  function webrtc(alive, onLost) {
    return new Promise(function (resolve, reject) {
      var finished = false;
      var guard = null;
      var lostTimer = null;

      function fail(msg, kind) {
        if (finished) return;
        finished = true;
        if (guard) clearTimeout(guard);
        if (lostTimer) clearTimeout(lostTimer);
        if (pc) { try { pc.close(); } catch (e) {} pc = null; }
        if (sessionUrl) {
          try { fetch(sessionUrl, { method: 'DELETE', keepalive: true }); } catch (e) {}
          sessionUrl = null;
        }
        var err = new Error(msg);
        err.kind = kind || 'error';
        reject(err);
      }

      function aborted() { fail('已取消', 'aborted'); }

      function done() {
        if (finished) return;
        finished = true;
        if (guard) clearTimeout(guard);
        resolve();
      }

      if (!alive()) return aborted();

      var conn = new RTCPeerConnection({ iceServers: [], bundlePolicy: 'max-bundle' });
      pc = conn;
      /* 骨架叠加层（_live_overlay_js.html）的约定：拿这个全局读 getStats 估延迟 */
      root.__fgLivePc = conn;
      root.__fgLiveMode = 'webrtc';

      conn.ontrack = function (ev) {
        if (video) video.srcObject = ev.streams[0];
        done();
      };
      conn.onconnectionstatechange = function () {
        if (pc !== conn) return;            /* 已被 stop()/新一轮起播替换 */
        var s = conn.connectionState;
        if (s === 'failed') {
          if (finished) { if (onLost) onLost('ICE 连接断开'); }
          else fail('ICE 连接失败');
          return;
        }
        if (s === 'disconnected') {
          /* disconnected 常常能自己恢复，给 4 秒再看一眼，别一抖就换 HLS */
          if (lostTimer) return;
          lostTimer = setTimeout(function () {
            lostTimer = null;
            if (pc !== conn) return;
            if (finished) { if (onLost) onLost('ICE 连接断开'); }
            else fail('ICE 断开');
          }, 4000);
          return;
        }
        if (s === 'connected' && lostTimer) {
          clearTimeout(lostTimer);
          lostTimer = null;
        }
      };

      conn.addTransceiver('video', { direction: 'recvonly' });
      guard = setTimeout(function () { fail('超时'); }, 6000);

      conn.createOffer()
        .then(function (offer) {
          if (!alive()) return Promise.reject(Object.assign(new Error('已取消'), { kind: 'aborted' }));
          return conn.setLocalDescription(offer);
        })
        .then(function () {
          if (!alive()) return Promise.reject(Object.assign(new Error('已取消'), { kind: 'aborted' }));
          return new Promise(function (res) {
            if (conn.iceGatheringState === 'complete') return res();
            conn.addEventListener('icegatheringstatechange', function () {
              if (conn.iceGatheringState === 'complete') res();
            });
            setTimeout(res, 2000);
          });
        })
        .then(function () {
          if (!alive()) return Promise.reject(Object.assign(new Error('已取消'), { kind: 'aborted' }));
          return fetch(WHEP_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/sdp' },
            body: conn.localDescription.sdp
          });
        })
        .then(function (r) {
          if (r.status === 404) {
            /* 这条路径上没有发布者 —— 不是网络问题，别回落，也别进冷却 */
            throw Object.assign(new Error('未连接'), { kind: 'not_connected' });
          }
          if (r.status !== 201) throw new Error('WHEP ' + r.status);
          sessionUrl = r.headers.get('Location');
          return r.text();
        })
        .then(function (answer) {
          if (!alive()) return Promise.reject(Object.assign(new Error('已取消'), { kind: 'aborted' }));
          return conn.setRemoteDescription({ type: 'answer', sdp: answer });
        })
        .then(done)
        .catch(function (err) { fail(err.message || 'WebRTC 失败', err.kind); });
    });
  }

  /* ── HLS（回落）────────────────────────────────────────────── */
  function loadHlsJs() {
    if (window.Hls) return Promise.resolve();
    return new Promise(function (resolve, reject) {
      var s = document.createElement('script');
      /* 没有超时的话，脚本请求挂住 = play() 的 Promise 永远不 settle，
         按钮卡死在禁用的"连接中…"，界面上连重试入口都没有。 */
      var timer = setTimeout(function () { reject(new Error('hls.js 加载超时')); }, 8000);
      s.src = HLS_JS;
      s.onload = function () { clearTimeout(timer); resolve(); };
      s.onerror = function () { clearTimeout(timer); reject(new Error('hls.js 加载失败')); };
      document.head.appendChild(s);
    });
  }

  function hlsPlay(alive) {
    return loadHlsJs().then(function () {
      if (!alive()) return Promise.reject(Object.assign(new Error('已取消'), { kind: 'aborted' }));
      if (window.Hls && window.Hls.isSupported()) {
        return new Promise(function (resolve, reject) {
          hls = new window.Hls({ lowLatencyMode: true, liveSyncDurationCount: 3 });
          var settled = false;
          var guard = setTimeout(function () {
            if (settled) return;
            settled = true;
            reject(new Error('HLS 超时'));
          }, 8000);
          hls.on(window.Hls.Events.MANIFEST_PARSED, function () {
            if (settled) return;
            settled = true;
            clearTimeout(guard);
            resolve();
          });
          hls.on(window.Hls.Events.ERROR, function (_e, data) {
            if (!data || !data.fatal) return;
            if (!settled) {
              settled = true;
              clearTimeout(guard);
              reject(new Error('HLS ' + (data.details || data.type)));
              return;
            }
            if (data.type === window.Hls.ErrorTypes.MEDIA_ERROR) {
              try { hls.recoverMediaError(); return; } catch (e) {}
            }
            if (!alive()) return;
            try { hls.destroy(); } catch (e) {}
            hls = null;
            setBanner('画面中断', '推流端断了或网络不稳，点「重试」重新连接。');
            setDot('播放中断', 'is-error');
            setBtn('重试', false);
            stop();
          });
          hls.loadSource(HLS_PLAYLIST);
          hls.attachMedia(video);
        });
      }
      if (video && video.canPlayType('application/vnd.apple.mpegurl')) {
        /* Safari 原生 HLS：没有 MANIFEST_PARSED 事件，只能先显示再等它自己出画 */
        video.src = HLS_PLAYLIST;
        return Promise.resolve();
      }
      throw new Error('当前浏览器不支持播放');
    });
  }

  /* ── 起播 ─────────────────────────────────────────────────── */
  function play() {
    if (busy || playing) return Promise.resolve();
    busy = true;
    var myGen = ++gen;
    var alive = function () { return myGen === gen; };
    setBtn('连接中…', true);

    function fallback(reason) {
      /* HLS 的延迟是分片级的（2–3 s），叠加层据此改用 2 s 基准 + 手动微调 */
      root.__fgLivePc = null;
      root.__fgLiveMode = 'hls';
      return hlsPlay(alive).then(function () {
        if (!alive()) return;
        showVideo();
        setDot('播放中（HLS）', 'is-live');
      }).catch(function (err) {
        if (!alive() || (err && err.kind === 'aborted')) return;
        setBanner('画面打不开', (reason ? reason + '；' : '') + (err.message || '未知错误'));
        setDot('播放失败', 'is-error');
        setBtn('重试', false);
        stop();
      });
    }

    /* 连上以后 ICE 才断：收掉 WebRTC，就地换 HLS，别把画面留在黑屏 */
    function onLiveLost(why) {
      if (!alive()) return;
      markWebrtcFailed();
      playing = false;
      teardownMedia();
      busy = true;
      setBtn('连接中…', true);
      fallback(why).then(function () { if (alive()) busy = false; });
    }

    var chain;
    if (!window.RTCPeerConnection || webrtcRecentlyFailed()) {
      chain = fallback('');
    } else {
      chain = webrtc(alive, onLiveLost)
        .then(function () {
          if (!alive()) return;
          showVideo();
        })
        .catch(function (err) {
          if (!alive() || (err && err.kind === 'aborted')) return;
          if (err && err.kind === 'not_connected') {
            setBanner('摄像头未连接', '这个点位配置了摄像头，但目前没有画面推上来。');
            setDot('未连接', 'is-off');
            setBtn('重试', false);
            stop();
            return;
          }
          markWebrtcFailed();
          return fallback(err.message);
        });
    }
    return chain.then(function () { if (alive()) busy = false; });
  }

  /* ── 先问状态，再决定要不要发请求 ─────────────────────────── */
  function check() {
    if (!STREAM) {
      setBanner('未配置摄像头', '这个点位还没有绑定摄像头，管理员配置后才能看。');
      setDot('未配置', 'is-off');
      setBtn('播放', true);
      return Promise.resolve('unconfigured');
    }

    return fetch(STATUS_URL, { headers: { 'Accept': 'application/json' } })
      .then(function (r) {
        if (r.status === 404) return { state: 'not_found' };
        if (!r.ok) return { state: 'unknown' };
        return r.json();
      })
      .then(function (d) {
        var st = (d && d.state) || 'unknown';
        if (st === 'online') {
          setDot('有信号', 'is-live');
          setBtn('播放', false);
          /* 进页面就播：用户要的是"点开点位就出现推流"。
             只播一路，带宽是 4.78Mbps 的硬限，别开第二个标签。
             后台标签页不自动起播——用户看不见，还占着上游的读者名额。
             AUTOPLAY=false（护士长病区画面）时连这个分支都不进：只把状态和
             「播放」按钮准备好，媒体流等人点了才建。 */
          if (AUTOPLAY && !autoStarted && !document.hidden) {
            autoStarted = true;
            play();
          }
          return st;
        }
        if (st === 'not_connected') {
          setBanner('摄像头未连接', '这个点位配置了摄像头，但目前没有画面推上来。');
          setDot('未连接', 'is-off');
          setBtn('重试', false);
          return st;
        }
        if (st === 'not_found') {
          /* 404 有两种含义：这个点位停用了，或者流没绑到本病区。
             都不能说成"没配置摄像头"——那会让人去查一个根本没配错的地方。 */
          setBanner('该点位不可用', '点位可能已停用，或画面还没有配置到本病区。请联系管理员。');
          setDot('不可用', 'is-off');
          setBtn('播放', true);
          return st;
        }
        setBanner('状态未知', '暂时连不上推流服务，稍后再试。');
        setDot('未知', 'is-error');
        setBtn('重试', false);
        return 'unknown';
      })
      .catch(function () {
        setBanner('状态未知', '暂时连不上服务，稍后再试。');
        setDot('未知', 'is-error');
        setBtn('重试', false);
        return 'unknown';
      });
  }

  if (btn) {
    btn.addEventListener('click', function () {
      if (playing || busy) {
        stop();
        resetUi('已停止', '');
        return;
      }
      if (!STREAM) return;
      play();
    });
  }

  /* 关页面/切后台必须收会话，否则云端会一直按"有观众"往外送流。
     切回来时重新问一次状态并自动起播——相当于"回到本页继续看"。 */
  window.addEventListener('pagehide', stop);
  /* 后退回到本页会从 bfcache 里整页还原：DOM 还是"播放中"的样子，
     但媒体连接早就在 pagehide 里断了。必须把界面也退回去。 */
  window.addEventListener('pageshow', function (e) {
    if (!e.persisted) return;
    stop();
    resetUi('已停止', '');
    autoStarted = false;
  });
  document.addEventListener('visibilitychange', function () {
    if (document.hidden) {
      stop();
      resetUi('已暂停', '切到后台时已停止播放。');
      autoStarted = false;
      return;
    }
    if (!STREAM || playing || busy) return;
    check();
  });

  /* 宫格删除某一格之前必须调它：DOM 一移走，RTCPeerConnection 和 HLS 实例
     不会自己消失，云端仍按「有观众」往外送流。 */
  root.__fgStop = stop;

  check();
}

window.fgInitCamera = fgInitCamera;

/* 单画面页：脚本加载时页面上已经有根了，直接绑。
   宫格页：格子由 head_camera.html 动态建，它建完自己调 fgInitCamera。 */
function fgBootCameras() {
  document.querySelectorAll('[data-camera-root]').forEach(function (el) {
    fgInitCamera(el);
  });
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', fgBootCameras);
} else {
  fgBootCameras();
}
