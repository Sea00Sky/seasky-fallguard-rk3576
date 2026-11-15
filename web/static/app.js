(function () {
  'use strict';

  var body = document.body;
  var FG = (window.FG = window.FG || {});

  /* ------------------------------------------------------------------------
     0. 基础工具
     ------------------------------------------------------------------------ */
  function $(sel, root) {
    return (root || document).querySelector(sel);
  }

  function $$(sel, root) {
    return Array.prototype.slice.call((root || document).querySelectorAll(sel));
  }

  function uuid() {
    try {
      if (window.crypto && typeof window.crypto.randomUUID === 'function') {
        return window.crypto.randomUUID();
      }
      var bytes = new Uint8Array(16);
      window.crypto.getRandomValues(bytes);
      bytes[6] = (bytes[6] & 15) | 64;
      bytes[8] = (bytes[8] & 63) | 128;
      var hex = [];
      for (var i = 0; i < bytes.length; i += 1) {
        hex.push(bytes[i].toString(16).padStart(2, '0'));
      }
      return hex.slice(0, 4).join('') + '-' + hex.slice(4, 6).join('') + '-' +
        hex.slice(6, 8).join('') + '-' + hex.slice(8, 10).join('') + '-' +
        hex.slice(10).join('');
    } catch (err) {
      return 'fg-' + Date.now() + '-' + Math.random().toString(16).slice(2);
    }
  }

  function toDate(value) {
    if (!value) return null;
    if (value instanceof Date) return isNaN(value.getTime()) ? null : value;
    var d = new Date(value);
    return isNaN(d.getTime()) ? null : d;
  }

  function pad2(n) {
    return String(n).padStart(2, '0');
  }

  function fmtClock(totalSeconds) {
    var s = Math.max(0, Math.floor(totalSeconds));
    var h = Math.floor(s / 3600);
    var m = Math.floor((s % 3600) / 60);
    var sec = s % 60;
    if (h > 0) return pad2(h) + ':' + pad2(m) + ':' + pad2(sec);
    return pad2(m) + ':' + pad2(sec);
  }

  function fmtDuration(totalSeconds) {
    var s = Math.max(0, Math.floor(totalSeconds));
    if (s < 60) return s + '秒';
    var m = Math.floor(s / 60);
    if (m < 60) return m + '分' + (s % 60) + '秒';
    return Math.floor(m / 60) + '小时' + (m % 60) + '分';
  }

  function relTime(date, now) {
    var diff = Math.max(0, Math.floor((now - date.getTime()) / 1000));
    if (diff < 60) return '刚刚';
    if (diff < 3600) return Math.floor(diff / 60) + ' 分钟前';
    if (diff < 86400) return Math.floor(diff / 3600) + ' 小时前';
    return Math.floor(diff / 86400) + ' 天前';
  }

  function escapeText(value) {
    return String(value == null ? '' : value);
  }

  function redirectLogin() {
    var next = window.location.pathname + window.location.search;
    window.location.assign('/login?next=' + encodeURIComponent(next) + '&reason=expired');
  }

  /* ------------------------------------------------------------------------
     1. Toast
     ------------------------------------------------------------------------ */
  var toastRoot = null;

  function ensureToastRoot() {
    if (!toastRoot) {
      toastRoot = $('#fg-toast-root');
      if (!toastRoot) {
        toastRoot = document.createElement('div');
        toastRoot.id = 'fg-toast-root';
        toastRoot.className = 'fg-toast-root';
        toastRoot.setAttribute('aria-live', 'polite');
        document.body.appendChild(toastRoot);
      }
    }
    return toastRoot;
  }

  function toast(message, type, duration) {
    var root = ensureToastRoot();
    var el = document.createElement('div');
    el.className = 'fg-toast' + (type ? ' fg-toast--' + type : '');
    el.setAttribute('role', type === 'error' ? 'alert' : 'status');
    var text = document.createElement('span');
    text.className = 'fg-toast__main';
    text.textContent = escapeText(message);
    el.appendChild(text);
    root.appendChild(el);
    var ms = duration || 4000;
    window.setTimeout(function () {
      if (el.parentNode) el.parentNode.removeChild(el);
    }, ms);
  }

  FG.toast = toast;

  /* ------------------------------------------------------------------------
     2. 声音：AudioContext 单音 880Hz / 0.18s
     ------------------------------------------------------------------------ */
  var SOUND_KEY = 'fg_sound';
  var audioCtx = null;
  var audioUnlocked = false;

  function soundEnabled() {
    try {
      return window.localStorage.getItem(SOUND_KEY) !== '0';
    } catch (err) {
      return true;
    }
  }

  function ensureAudio() {
    var Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return null;
    if (!audioCtx) {
      try {
        audioCtx = new Ctx();
      } catch (err) {
        return null;
      }
    }
    if (audioCtx.state === 'suspended') {
      audioCtx.resume().catch(function () {});
    }
    audioUnlocked = true;
    return audioCtx;
  }

  function beep() {
    if (!soundEnabled()) return;
    var ctx = audioCtx;
    if (!ctx || ctx.state !== 'running') return;
    try {
      var now = ctx.currentTime;
      var osc = ctx.createOscillator();
      var gain = ctx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(880, now);
      gain.gain.setValueAtTime(0.0001, now);
      gain.gain.exponentialRampToValueAtTime(0.18, now + 0.01);
      gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.18);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start(now);
      osc.stop(now + 0.2);
    } catch (err) {
      /* 音频失败不影响业务 */
    }
  }

  FG.beep = beep;
  FG.ensureAudio = ensureAudio;

  document.addEventListener('pointerdown', function () {
    ensureAudio();
  }, { once: true, capture: true });

  /* ------------------------------------------------------------------------
     3. BroadcastChannel 选主：只有一个标签页发声
     ------------------------------------------------------------------------ */
  /* 每个标签页一个随机 id（不写 sessionStorage：复制标签页会共享 sessionStorage，
     导致两个标签都认为自己是主标签、同时发声） */
  var tabId = uuid();

  var bus = null;
  var peers = {};

  function prunePeers() {
    var now = Date.now();
    Object.keys(peers).forEach(function (id) {
      if (now - peers[id] > 6000) delete peers[id];
    });
  }

  function isLeader() {
    prunePeers();
    var leader = tabId;
    Object.keys(peers).forEach(function (id) {
      if (id < leader) leader = id;
    });
    return leader === tabId;
  }

  function initBus() {
    if (!('BroadcastChannel' in window)) return;
    try {
      bus = new BroadcastChannel('fg_alert_bus');
      bus.onmessage = function (event) {
        var msg = event.data || {};
        if (msg.type === 'beat' || msg.type === 'hello') {
          peers[msg.id] = Date.now();
          if (msg.type === 'hello' && bus) {
            bus.postMessage({ type: 'beat', id: tabId, t: Date.now() });
          }
        } else if (msg.type === 'bye') {
          delete peers[msg.id];
        }
      };
      bus.postMessage({ type: 'hello', id: tabId });
      window.setInterval(function () {
        if (bus) bus.postMessage({ type: 'beat', id: tabId, t: Date.now() });
      }, 2000);
      window.addEventListener('pagehide', function () {
        if (bus) bus.postMessage({ type: 'bye', id: tabId });
      });
    } catch (err) {
      bus = null;
    }
  }

  /* ------------------------------------------------------------------------
     4. 连接状态横幅（48px，不可关闭）
     ------------------------------------------------------------------------ */
  var liveBanner = null;

  function ensureLiveBanner() {
    if (liveBanner) return liveBanner;
    liveBanner = document.createElement('div');
    liveBanner.className = 'fg-banner fg-banner--danger';
    liveBanner.id = 'fg-live-banner';
    liveBanner.setAttribute('role', 'alert');
    liveBanner.hidden = true;
    var text = document.createElement('span');
    text.className = 'fg-banner__text';
    text.textContent = '实时连接已断，新告警可能未送达。正在自动重连，也可手动刷新页面。';
    liveBanner.appendChild(text);
    var main = $('.fg-main');
    var topbar = $('.fg-topbar');
    if (main && topbar && topbar.parentNode === main) {
      main.insertBefore(liveBanner, topbar.nextSibling);
    } else if (main) {
      main.insertBefore(liveBanner, main.firstChild);
    }
    return liveBanner;
  }

  function setConnection(state) {
    var banner = ensureLiveBanner();
    if (!banner) return;
    banner.hidden = state !== 'offline';
    var statuses = $$('[data-connection-status]');
    statuses.forEach(function (el) {
      el.setAttribute('data-state', state);
    });
  }

  /* ------------------------------------------------------------------------
     5. 标题闪烁
     ------------------------------------------------------------------------ */
  var baseTitle = document.title;
  var titleTimer = null;

  function flashTitle() {
    if (!document.hidden) return;
    if (titleTimer) return;
    var on = false;
    titleTimer = window.setInterval(function () {
      on = !on;
      document.title = on ? '【新告警】' + baseTitle : baseTitle;
    }, 1000);
  }

  function stopFlash() {
    if (titleTimer) {
      window.clearInterval(titleTimer);
      titleTimer = null;
    }
    document.title = baseTitle;
  }

  window.addEventListener('focus', stopFlash);

  var streamUrl = body.getAttribute('data-stream-url') || '';
  var eventSource = null;
  var sseOpen = false;
  var pollTimer = null;
  var pollIntervalMs = 5000;
  var polling = false;
  var pendingCount = null;
  var PENDING_QUERY = '/api/v1/events?disposition=PENDING&test=0&size=1';
  var lastNotifyAt = 0;
  var refreshTimer = null;
  var refreshing = false;
  /* 后退落在一次在途刷新里时排队，等那次收尾再刷（见 initWorkbench 的 popstate）。 */
  var pendingReload = false;

  function notifyNewAlert() {
    var now = Date.now();
    if (now - lastNotifyAt < 3000) return;
    lastNotifyAt = now;
    if (isLeader()) beep();
    flashTitle();
  }

  /* ------------------------------------------------------------------------
     7. 跌倒告警弹窗

     三条设计取舍，答辩可以直接照着讲：

     1) 为什么不刷新页面也要弹：告警走 SSE `event.created` 推送，页面不动。
        所以弹窗内容全部由 JS 从推送 payload 填充（模板里只留骨架）。
     2) 为什么按 document.hidden 分流：
        - 页面可见 → 页内模态框，能立刻看到快照、点位、置信度；
        - 页面在后台 → 页内弹窗根本看不见。
           桌面通知整条删除：Notification API 只在安全上下文可用，
          本站是 `http://` + IP，浏览器对明文来源强制 `permission=denied`（实测全新
          浏览器配置即 denied，站点设置里也没得改）⇒ 这个能力在现网永远点不亮。
          所以后台分支从此什么都不做：后台到达的告警不弹窗、也不补弹，
          只靠提示音 / 标题闪烁 / 侧边栏红点。哪天上了 HTTPS 再从 git 历史捞回来。
     3) 为什么不叠加：连续跌倒时队列最多留 3 条，弹一条、关一条、再弹下一条；
        逐条确认比 5 个弹窗摞在一起安全。

      快照比事件晚到：`has_snapshot` 在 SSE payload 里恒为 false（上传器几秒后
        才把图推到云端），所以缩略图要轮询重试，不能只试一次。
     ------------------------------------------------------------------------ */
  var ALARM_MAX_QUEUE = 3;         // 队列上限，超出丢最旧的
  var ALARM_THUMB_TRIES = 10;      // 快照重试次数（1.5s × 10 ≈ 15s）
  var ALARM_THUMB_GAP = 1500;

  var alarmQueue = [];
  var alarmShowing = null;         // 当前展示的 event_uid，兼作「弹窗是否打开」标志

  function alarmNode(id) {
    return document.getElementById(id);
  }

  function alarmSetText(id, text) {
    var el = alarmNode(id);
    if (el) el.textContent = text;
  }

  function alarmDetailUrl(uid) {
    var role = body.getAttribute('data-role') || 'nurse';
    var base = role === 'head_nurse' ? '/head/events/' : '/alerts/';
    return base + encodeURIComponent(uid);
  }

  function alarmSevLabel(sev) {
    if (sev === 'S2') return 'S2 重要';
    if (sev === 'S3') return 'S3 提示';
    return 'S1 紧急';
  }

  function alarmClock(value) {
    var d = toDate(value);
    if (!d) return '—';
    return pad2(d.getMonth() + 1) + '-' + pad2(d.getDate()) + ' ' +
      pad2(d.getHours()) + ':' + pad2(d.getMinutes());
  }

  /* 模型原始分，不是「置信度」，也不给百分比。
     板端 INT8 模型的输出概率没有做校准（详情页的 prob_text 宏同样写「未校准」），
     把它渲染成「87%」等于替模型承诺了一个它没给出的可信度。
     全站统一这个名字：模型原始分。 */
  function alarmProb(value) {
    var n = Number(value);
    if (!isFinite(n)) return '—';
    return n.toFixed(2) + '（未校准）';
  }

  function loadAlarmThumb(uid, attempt) {
    if (alarmShowing !== uid) return;        // 已切到下一条，放弃这次重试
    var img = alarmNode('fg-alarmpop-img');
    var ph = alarmNode('fg-alarmpop-ph');
    if (!img || !ph) return;

    var probe = new Image();
    probe.onload = function () {
      if (alarmShowing !== uid) return;
      img.src = probe.src;
      img.hidden = false;
      ph.hidden = true;
    };
    probe.onerror = function () {
      if (alarmShowing !== uid) return;
      if (attempt >= ALARM_THUMB_TRIES) {
        var spinner = ph.querySelector('.fg-alarmpop__spinner');
        if (spinner) spinner.hidden = true;
        alarmSetText('fg-alarmpop-phtext', '暂无快照（板端未上传或已过期）');
        return;
      }
      window.setTimeout(function () {
        loadAlarmThumb(uid, attempt + 1);
      }, ALARM_THUMB_GAP);
    };
    /* 缩略图路由是 max-age=60，重试期间会被浏览器缓存住，必须带时间戳破缓存 */
    probe.src = '/api/v1/events/' + encodeURIComponent(uid) +
      '/snapshot/thumb?t=' + Date.now();
  }

  /* ---- 渲染 / 开关 ---- */
  function renderAlarm(data) {
    var root = alarmNode('fg-alarmpop');
    if (!root) return;
    var sev = (data.severity === 'S2' || data.severity === 'S3') ? data.severity : 'S1';
    root.setAttribute('data-sev', sev);

    var tag = alarmNode('fg-alarmpop-sev');
    if (tag) {
      tag.className = 'fg-tag fg-tag--' + sev.toLowerCase();
      tag.textContent = alarmSevLabel(sev);
    }

    alarmSetText('fg-alarmpop-title', sev === 'S3' ? '疑似跌倒，请确认' : '检测到跌倒');
    alarmSetText('fg-alarmpop-loc', data.location || '未命名点位');
    alarmSetText('fg-alarmpop-time', alarmClock(data.received_at));
    alarmSetText('fg-alarmpop-prob', alarmProb(data.prob_fall));
    alarmSetText('fg-alarmpop-person',
      (data.person_id === null || data.person_id === undefined)
        ? '—' : '人员#' + data.person_id);

    var go = alarmNode('fg-alarmpop-go');
    if (go) go.href = alarmDetailUrl(data.event_uid);

    var img = alarmNode('fg-alarmpop-img');
    var ph = alarmNode('fg-alarmpop-ph');
    if (img) {
      img.hidden = true;
      img.removeAttribute('src');
    }
    if (ph) {
      ph.hidden = false;
      var spinner = ph.querySelector('.fg-alarmpop__spinner');
      if (spinner) spinner.hidden = false;
      alarmSetText('fg-alarmpop-phtext', '快照上传中…');
    }

  }

  function updateAlarmQueueHint() {
    var el = alarmNode('fg-alarmpop-queue');
    if (!el) return;
    if (alarmQueue.length > 0) {
      el.textContent = '还有 ' + alarmQueue.length + ' 条新告警待确认';
      el.hidden = false;
    } else {
      el.textContent = '';
      el.hidden = true;
    }
  }

  function nextAlarm() {
    if (alarmShowing) return;               // 已经有弹窗开着
    if (!alarmQueue.length) return;
    var data = alarmQueue.shift();
    var root = alarmNode('fg-alarmpop');
    if (!root) return;
    alarmShowing = data.event_uid;
    renderAlarm(data);
    updateAlarmQueueHint();
    openRoot(root);
    var card = root.querySelector('.fg-alarmpop__card');
    if (card && card.focus) {
      try { card.focus({ preventScroll: true }); } catch (err) { card.focus(); }
    }
    loadAlarmThumb(data.event_uid, 0);
  }

  function onAlarmClosed() {
    alarmShowing = null;
    if (alarmQueue.length) {
      window.setTimeout(nextAlarm, 900);    // 留一点缓冲，别一关就蹦出下一条
    }
  }

  function enqueueAlarm(data) {
    if (!data || !data.event_uid) return;
    if (alarmShowing === data.event_uid) return;
    for (var i = 0; i < alarmQueue.length; i += 1) {
      if (alarmQueue[i].event_uid === data.event_uid) return;
    }
    alarmQueue.push(data);
    while (alarmQueue.length > ALARM_MAX_QUEUE) alarmQueue.shift();
    if (!alarmShowing) {
      nextAlarm();
    }
    updateAlarmQueueHint();
  }

  /* SSE 回调入口：后台标签页直接丢弃。页内弹窗在后台不可见，堆积到回前台
     会一次性弹出多条过时告警；漏掉的那条由提示音 / 标题闪烁 / 红点兜底。 */
  function dispatchAlarm(data) {
    if (!data || !data.event_uid) return;
    if (document.hidden) return;
    enqueueAlarm(data);
  }

  function initAlarmPopup() {
    var root = alarmNode('fg-alarmpop');
    if (!root) return;
    $$('[data-alarmpop-dismiss]', root).forEach(function (el) {
      el.addEventListener('click', function () {
        closeRoot(root);
      });
    });
  }

  /* 待处置计数（属于 §6 推送链路的尾巴，不单列一节） */
  function setPendingCount(next) {
    var value = Number(next);
    if (!isFinite(value)) return;
    var previous = pendingCount;
    pendingCount = value;
    $$('[data-pending-count]').forEach(function (el) {
      el.textContent = value > 99 ? '99+' : String(value);
      el.hidden = value <= 0;
    });
    if (previous !== null && value > previous) notifyNewAlert();
  }

  function bumpPending(delta) {
    if (pendingCount === null) return;
    setPendingCount(Math.max(0, pendingCount + delta));
  }

  function handleStreamEvent(event, isCreated) {
    var data = {};
    try {
      data = JSON.parse(event.data || '{}');
    } catch (err) {
      return;
    }
    if (isCreated && data.disposition_state === 'PENDING' &&
        data.detection_state !== 'RETRACTED') {
      bumpPending(1);
      notifyNewAlert();
      dispatchAlarm(data);
    }
    if (data.has_clip && data.event_uid) {
      upgradeClipSection(String(data.event_uid));
    }
    scheduleRefresh();
  }

  /* 片段切好后（worker 广播 has_clip）把「生成中」占位就地换成播放器。
     不走整页刷新：详情页上可能正有人在填处置表单，刷新会把输入冲掉。 */
  function upgradeClipSection(uid) {
    var section = document.getElementById('fg-clip-section');
    if (!section) return;
    if (section.getAttribute('data-event-uid') !== uid) return;
    if (section.getAttribute('data-clip-status') === 'READY') return;

    var video = document.createElement('video');
    video.className = 'fg-clip';
    video.controls = true;
    video.preload = 'metadata';
    video.playsInline = true;
    video.src = '/api/v1/events/' + encodeURIComponent(uid) + '/clip';

    var foot = document.createElement('div');
    foot.className = 'fg-small fg-muted fg-mt-2';
    foot.textContent = '告警前 5 秒 + 后 5 秒 · 云端从推流录制里切出 · 不重编码，画质与实时画面一致';

    section.innerHTML = '';
    section.appendChild(video);
    section.appendChild(foot);
    section.setAttribute('data-clip-status', 'READY');
  }

  function connectStream() {
    if (!('EventSource' in window)) {
      setConnection('offline');
      startPolling(5000);
      return;
    }
    try {
      eventSource = new EventSource(streamUrl, { withCredentials: true });
    } catch (err) {
      setConnection('offline');
      startPolling(5000);
      return;
    }
    eventSource.addEventListener('open', function () {
      sseOpen = true;
      setConnection('online');
      startPolling(30000);
    });
    eventSource.addEventListener('hello', function () {
      sseOpen = true;
      setConnection('online');
    });
    eventSource.addEventListener('event.created', function (event) {
      handleStreamEvent(event, true);
    });
    eventSource.addEventListener('event.updated', function (event) {
      handleStreamEvent(event, false);
    });
    eventSource.onerror = function () {
      sseOpen = false;
      setConnection('offline');
      startPolling(5000);
    };
  }

  function startPolling(ms) {
    pollIntervalMs = ms;
    if (pollTimer) {
      window.clearInterval(pollTimer);
      pollTimer = null;
    }
    if (!ms || ms <= 0) return;
    pollTimer = window.setInterval(pollOnce, ms);
    pollOnce();
  }

  function extractTotal(payload) {
    if (!payload || typeof payload !== 'object') return null;
    if (typeof payload.total === 'number') return payload.total;
    if (typeof payload.count === 'number') return payload.count;
    var list = payload.items || payload.events || payload.data;
    if (Array.isArray(list)) return list.length;
    return null;
  }

  function pollOnce() {
    if (document.hidden || polling) return;
    /* 没有实时流的角色（信息科管理员）不轮询：/api/v1/events 对他是 403，
       每切一次标签页就打一次越权接口、还往审计里写一条 DENIED。 */
    if (!body.hasAttribute('data-authed') || !streamUrl) return;
    polling = true;
    var request = fetch(PENDING_QUERY, {
      method: 'GET',
      credentials: 'same-origin',
      cache: 'no-store',
      headers: { 'Accept': 'application/json', 'X-Requested-With': 'fg-poll' }
    }).then(function (res) {
      if (res.status === 401) {
        redirectLogin();
        return null;
      }
      if (!res.ok) return null;
      return res.json().catch(function () { return null; });
    }).then(function (payload) {
      var total = extractTotal(payload);
      if (total !== null) setPendingCount(total);
    }).catch(function () {
      setConnection('offline');
    }).then(function () {
      polling = false;
      if (!sseOpen) return refreshRegions();
      return null;
    });
    return request;
  }

  function refreshPendingCount() {
    if (!body.hasAttribute('data-authed') || !streamUrl) return;
    return fetch(PENDING_QUERY, {
      method: 'GET',
      credentials: 'same-origin',
      cache: 'no-store',
      headers: { 'Accept': 'application/json', 'X-Requested-With': 'fg-poll' }
    }).then(function (res) {
      if (!res.ok) return null;
      return res.json().catch(function () { return null; });
    }).then(function (payload) {
      var total = extractTotal(payload);
      if (total !== null) setPendingCount(total);
    }).catch(function () { /* 角标刷新失败不弹错：下一轮轮询会补上 */ });
  }

  function scheduleRefresh() {
    if (refreshTimer) return;
    refreshTimer = window.setTimeout(function () {
      refreshTimer = null;
      refreshRegions();
    }, 800);
  }

  /* 工作区里正开着一张没提交的表单（处置结果 / 误报原因）时不能替换它：
     自动刷新会把护士刚选好的那一项冲掉。焦点守卫只挡得住「正在输入」，
     挡不住「已经选好、手还没离开鼠标」那一瞬。 */
  function regionBusy(el) {
    return !!el.querySelector('form[data-action-form]:not([hidden]), .fg-modal-root:not([hidden])');
  }

  /* 记住焦点在哪，换完 DOM 再找回来。
     用 Tab 走到第 3 行按 Enter，那一行会被 replaceWith 换掉 ⇒ 焦点掉回 body，
     再按 Tab 又从页面顶部重来。键盘用户等于每次切人都被踢回起点。 */
  function focusAnchor(el) {
    if (!el || el === document.body || !el.closest) return null;
    var row = el.closest('[data-workspace-open]');
    if (row) {
      var m = /focus=([^&]+)/.exec(row.getAttribute('href') || '');
      if (m) return { kind: 'row', uid: decodeURIComponent(m[1]) };
    }
    if (el.id) return { kind: 'id', id: el.id };
    var act = el.closest('[data-action]');
    if (act) return { kind: 'action', action: act.getAttribute('data-action') };
    return null;
  }

  function restoreFocus(anchor) {
    if (!anchor) return;
    var el = null;
    if (anchor.kind === 'row') {
      $$('[data-workspace-open]').forEach(function (a) {
        if (el) return;
        var m = /focus=([^&]+)/.exec(a.getAttribute('href') || '');
        if (m && decodeURIComponent(m[1]) === anchor.uid) el = a;
      });
      /* 那一行可能已经不在当前分段了（刚接完警就离开「待接警」）——
         退回到高亮行，键盘位置至少还在这张队列里。 */
      if (!el) el = $('[data-workspace-open][aria-current="true"]');
    } else if (anchor.kind === 'id') {
      el = document.getElementById(anchor.id);
    } else if (anchor.kind === 'action') {
      el = $('[data-action="' + anchor.action + '"]');
    }
    if (el && el.focus) el.focus({ preventScroll: true });
  }

  /* 把服务端渲染的同一批 region 换进来。keys 为空 = 全部换。
     force=true：调用方是用户主动切页（点队列行 / 切 Tab）。这时必须换，
     否则会「左栏、地址栏都换了，右栏还停在旧事件」——用户点的那一下被静默吞掉。
     regionBusy 只该保护自动刷新（SSE / 轮询），不该拦用户。 */
  function swapRegions(doc, keys, force) {
    var keepScroll = null;
    var paneBefore = $('.fg-work__body');
    if (paneBefore && !force) keepScroll = paneBefore.scrollTop;
    /* 一屏布局后队列列表自己滚。同段内换行/刷新要把滚动位置带过去，
       否则点了第 12 行，列表重渲染后弹回顶部，选中行直接看不见。
       换段（Tab）则归顶：新段不是同一批行，沿用旧偏移没有意义。 */
    var qBodyBefore = $('[data-refresh-region="worklist"] .fg-panel__body');
    var qScroll = qBodyBefore ? qBodyBefore.scrollTop : 0;
    /* 病区总览一屏化（ ）后待处置队列也自己滚。同理：不带走滚动位置，
       每 30 秒自动刷新一次就把正在读的那一行弹回顶部。 */
    var ovQueueBefore = $('[data-refresh-region="pending-followup"] .fg-queue');
    var ovScroll = ovQueueBefore ? ovQueueBefore.scrollTop : 0;
    var tableScroll = {};
    $$('[data-refresh-region]').forEach(function (el) {
      var w = el.querySelector('.fg-table-wrap');
      if (w && w.scrollTop) {
        tableScroll[el.getAttribute('data-refresh-region')] = w.scrollTop;
      }
    });
    var qTabBefore = $('[data-refresh-region="worklist"] .fg-tab.is-active');
    var qTabKey = qTabBefore ? qTabBefore.getAttribute('href') : null;
    var active = document.activeElement;
    var anchor = (active && active.closest('[data-refresh-region]')) ? focusAnchor(active) : null;
    $$('[data-refresh-region]').forEach(function (el) {
      var key = el.getAttribute('data-refresh-region');
      if (keys && keys.indexOf(key) < 0) return;
      if (!force && regionBusy(el)) return;
      /* 弹窗已经被搬到 body（见 relocateModals），regionBusy 查不到它 ——
         但工作区换掉就等于把弹窗背后的那条事件换走了。开着弹窗时不自动换工作区。 */
      if (!force && key === 'workspace' && $$('.fg-modal-root:not([hidden])').length) return;
      /* 片段正在播：整块替换会让 <video> 重建、从 0 重播。
         护士正看着回放时被拽回开头，比看到旧数据更糟。 */
      if (!force && key === 'workspace') {
        var playing = $$('[data-refresh-region="workspace"] video').some(function (v) {
          return !v.paused && !v.ended;
        });
        if (playing) return;
      }
      var fresh = doc.querySelector('[data-refresh-region="' + key + '"]');
      if (fresh) el.replaceWith(fresh);
    });
    scanDynamic();
    /* 自动刷新时别把正在看时间线的人拽回顶部；主动切行才归零（loadWorkspace 里做）。 */
    if (keepScroll) {
      var paneAfter = $('.fg-work__body');
      if (paneAfter) paneAfter.scrollTop = keepScroll;
    }
    var qTabAfter = $('[data-refresh-region="worklist"] .fg-tab.is-active');
    var qTabKeyAfter = qTabAfter ? qTabAfter.getAttribute('href') : null;
    if (qScroll && qTabKey !== null && qTabKey === qTabKeyAfter) {
      var qBodyAfter = $('[data-refresh-region="worklist"] .fg-panel__body');
      if (qBodyAfter) qBodyAfter.scrollTop = qScroll;
    }
    if (ovScroll) {
      var ovQueueAfter = $('[data-refresh-region="pending-followup"] .fg-queue');
      if (ovQueueAfter) ovQueueAfter.scrollTop = ovScroll;
    }
    Object.keys(tableScroll).forEach(function (key) {
      var w = $('[data-refresh-region="' + key + '"] .fg-table-wrap');
      if (w) w.scrollTop = tableScroll[key];
    });
    restoreFocus(anchor);
  }

  /* 响应拿回来的不是本站页面（典型：会话过期，_guard 302 到 /login，
     fetch 跟随重定向后 200 但 HTML 里没有任何 region）。这种情况必须当失败：
     否则点行只改地址栏、右栏纹丝不动，人还以为是自己点错了。 */
  function looksLikeAppPage(doc) {
    return !!doc.querySelector('[data-refresh-region]');
  }

  /* 把「这一屏正在显示哪条」钉进地址栏。
     没有这一步，URL 里只有 ?disposition=PENDING：动作后自动刷新会重新请求它，
     服务端按 rows[0] 重算 —— 刚接完警那条已经离开待接警，工作区就跳到另一条，
     「接完警下一步在原地」当场失效。 */
  function pinFocusInUrl(url) {
    var pane = $('[data-refresh-region="workspace"]');
    var uid = pane && pane.getAttribute('data-focus-uid');
    if (!uid || url.searchParams.get('focus')) return url;
    url.searchParams.set('focus', uid);
    return url;
  }

  function refreshRegions() {
    if (refreshing) return Promise.resolve();
    var regions = $$('[data-refresh-region]');
    if (!regions.length) return Promise.resolve();
    /* 用户正在筛选框里输入时不要替换 DOM，避免焦点丢失 */
    var active = document.activeElement;
    if (active && active.closest('[data-refresh-region]') &&
        /^(INPUT|TEXTAREA|SELECT)$/.test(active.tagName)) {
      return Promise.resolve();
    }
    refreshing = true;
    var url = new URL(window.location.href);
    url.searchParams.set('_fg', String(Date.now()));
    /* 发起时的地址：响应回来若地址已被 popstate / 点行改掉，这批 HTML 就是过期的。 */
    var issuedFor = window.location.pathname + window.location.search;
    return fetch(url.toString(), {
      method: 'GET',
      credentials: 'same-origin',
      cache: 'no-store',
      headers: { 'Accept': 'text/html', 'X-Requested-With': 'fg-refresh' }
    }).then(function (res) {
      if (res.status === 401) {
        redirectLogin();
        return null;
      }
      if (!res.ok) return null;
      if (res.redirected && /\/login(\?|$)/.test(new URL(res.url).pathname)) {
        redirectLogin();
        return null;
      }
      return res.text();
    }).then(function (html) {
      if (!html) return;
      if (window.location.pathname + window.location.search !== issuedFor) return;
      var doc = new DOMParser().parseFromString(html, 'text/html');
      if (!looksLikeAppPage(doc)) {
        redirectLogin();
        return;
      }
      swapRegions(doc);
    }).catch(function () {
      /* 刷新失败保留旧内容，不白屏 */
    }).then(function () {
      refreshing = false;
      if (pendingReload) {
        pendingReload = false;
        refreshRegions();
      }
    });
  }

  /* ------------------------------------------------------------------------
     9.5 工作台：左栏选人、右栏干活（ ）
     ------------------------------------------------------------------------
     点队列里的某一条 = 换工作区那一条。走 fetch + pushState，不整页刷新：
     护士一次值班要在一屏里切几十次，每次整页重载会把滚动位置和焦点全丢掉。
     无 JS / 请求失败时回落到普通跳转——行本身就是个 <a href>，服务端照样渲染。 */
  function loadWorkspace(href) {
    /* 基准必须是当前页地址（window.location.href），不能是 origin。
       队列行的 href 是相对的 "?disposition=…&focus=…"，而 origin（"http://host"）
       的 path 是 "/"，`new URL('?x=1', origin)` 会解析成 "/?x=1" ——
       请求打到根路由，302 回 /nurse/today 且丢掉 query，于是每次点行拿回来的
       都是默认第一条，工作区看起来「点了没反应」。 线上就是这个 bug。 */
    var url = new URL(href, window.location.href);
    var target = new URL(url.toString());
    target.searchParams.set('_fg', String(Date.now()));
    /* 刷新正在进行时不要并行换内容：两次渲染谁后到谁赢，可能出现
       「队列已换成新选中，工作区还是旧的」。退回整页跳转，至少是对的。 */
    if (refreshing) {
      window.location.assign(url.toString());
      return;
    }
    refreshing = true;
    return fetch(target.toString(), {
      method: 'GET',
      credentials: 'same-origin',
      cache: 'no-store',
      headers: { 'Accept': 'text/html', 'X-Requested-With': 'fg-workspace' }
    }).then(function (res) {
      if (res.status === 401) {
        redirectLogin();
        return null;
      }
      if (!res.ok) throw new Error('HTTP ' + res.status);
      if (res.redirected && /\/login(\?|$)/.test(new URL(res.url).pathname)) {
        redirectLogin();
        return null;
      }
      return res.text();
    }).then(function (html) {
      if (!html) return;
      var doc = new DOMParser().parseFromString(html, 'text/html');
      /* 拿回来的不是工作台（会话过期 / 被重定向）就走下面的 catch 整页跳转，
         别用一份缺 region 的文档去做「成功」状的 pushState。 */
      if (!doc.querySelector('[data-refresh-region="workspace"]')) {
        throw new Error('not a workbench page');
      }
      /* force=true：这是用户主动点的那一行，必须换——哪怕工作区里正开着一张表单。
         半填内容会丢，但「点了没反应」更糟；表格本身只有单选 + 可选备注，重填成本低。
         kpi-row 一起换：白板副行「X 待接警 · Y 处理中」也随分段走，不换就会和 Tab 数字打架。
         device-strip 一起换（ ）：点位灯的筛选钮点下去之后，
         灯角那个「已筛选」状态和筛选条是同一份服务端渲染，不换就两边对不上。 */
      swapRegions(doc, ['worklist', 'workspace', 'kpi-row', 'device-strip'], true);
      /* 切 Tab 时 href 不带 focus，服务端补了默认第一条 —— 地址里也要补上，
         否则下一次自动刷新又会把工作区换成「那时」的第一条。 */
      var landed = pinFocusInUrl(new URL(url.toString()));
      window.history.pushState({ fgWorkspace: landed.toString() }, '', landed.toString());
      /* 换了人就把工作区滚回顶部——否则停在上一条的中间，看起来像没切换 */
      var pane = $('.fg-work__body');
      if (pane) pane.scrollTop = 0;
      /* 键盘用户：焦点跟着选中那条走，别掉回 body 从头 Tab 起。 */
      var row = $('[data-workspace-open][aria-current="true"]');
      if (row && row.focus) {
        row.focus({ preventScroll: true });
        /* 队列列表是内部滚动：选中行在可视区外时滚进来。
           页面本身不滚，scrollIntoView 只会动队列的滚动容器（窄屏下才会动页面）。 */
        if (row.scrollIntoView) row.scrollIntoView({ block: 'nearest', inline: 'nearest' });
      }
    }).catch(function () {
      window.location.assign(url.toString());
    }).then(function () {
      refreshing = false;
      if (pendingReload) {
        pendingReload = false;
        refreshRegions();
      }
    });
  }

  function initWorkbench() {
    if (!$('[data-refresh-region="workspace"]')) return;

    document.addEventListener('click', function (event) {
      var link = event.target.closest('[data-workspace-open]');
      if (!link) return;
      /* 中键 / Ctrl 点击等「新开标签」的意图要放行，别抢 */
      if (event.defaultPrevented || event.button !== 0 ||
          event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      loadWorkspace(link.getAttribute('href'));
    });

    window.addEventListener('popstate', function () {
      if (window.location.pathname !== '/nurse/today') return;
      if (refreshing) {
        pendingReload = true;
        return;
      }
      refreshRegions();
    });

    /* 首屏默认选中（URL 里没有 focus）也要写进地址：否则动作后的自动刷新
       重新请求当前 URL，服务端按 rows[0] 重算——刚接完警那条已离开待接警，
       工作区就跳到另一条去了，正好毁掉「接完警下一步在原地」。
       用 replaceState 而不是 302：点 Tab 之后按后退不会在两个 URL 之间打转。 */
    var pane = $('[data-refresh-region="workspace"]');
    var focusUid = pane && pane.getAttribute('data-focus-uid');
    if (focusUid && !new URL(window.location.href).searchParams.get('focus')) {
      var cur = new URL(window.location.href);
      cur.searchParams.set('focus', focusUid);
      window.history.replaceState({ fgWorkspace: cur.toString() }, '', cur.toString());
    }

  }

  /* 演示 / 排障：浏览器控制台执行 FG.demoAlarm() 就能预览弹窗，
     不用真让人摔一跤。传 {event_uid:'<真实事件号>'} 连真快照一起拉。
     直接走 enqueueAlarm，不经过 dispatchAlarm 的可见性分流 —— 用户主动触发
     就该看到结果，哪怕当前标签页在后台。 */
  FG.demoAlarm = function (overrides) {
    var data = {
      event_uid: 'demo-' + Date.now(),
      severity: 'S1',
      detection_state: 'CONFIRMED',
      disposition_state: 'PENDING',
      received_at: new Date().toISOString(),
      location: '3F 走廊东侧 · 摄像头 01',
      device_code: 'K7C-4FA-01',
      person_id: 3,
      prob_fall: 0.91,
      coverage: 0.88,
      has_snapshot: false
    };
    if (overrides) {
      Object.keys(overrides).forEach(function (key) {
        data[key] = overrides[key];
      });
    }
    enqueueAlarm(data);
    return data;
  };

  /* ------------------------------------------------------------------------
     8. 倒计时 / 相对时间 / 定时刷新
     ------------------------------------------------------------------------ */
  function relocateModals() {
    $$('.fg-modal-root').forEach(function (el) {
      /* 快照里新旧两份都在。新节点先挂上 body 时会把旧节点删掉，
         但 forEach 接着还会遍历到那个已经脱离文档的旧节点：它的 parentNode 是
         null（≠ body），于是按 id 查 body —— 查到的是刚挂上去的新节点，
         被删掉之后旧节点又被 appendChild 挂回 body。净效果是「换行后弹窗还绑着上一条事件」。
          线上实测到的就是这个：切到第 3 条，body 上两份弹窗 uid 全是旧事件。 */
      if (!el.isConnected) return;
      if (el.parentNode === document.body) return;
      if (el.id) {
        $$('body > .fg-modal-root[id="' + el.id + '"]').forEach(function (old) {
          if (old !== el) old.remove();
        });
      }
      document.body.appendChild(el);
    });
  }

  function scanDynamic() {
    relocateModals();
    tick();
  }

  function tick() {
    var now = Date.now();

    $$('[data-wait-since]').forEach(function (el) {
      var date = toDate(el.getAttribute('data-wait-since'));
      if (!date) {
        el.textContent = '—';
        return;
      }
      el.textContent = fmtClock((now - date.getTime()) / 1000);
    });

    $$('[data-relative-time]').forEach(function (el) {
      var date = toDate(el.getAttribute('data-relative-time'));
      if (!date) {
        el.textContent = '—';
        return;
      }
      var seconds = Math.max(0, (now - date.getTime()) / 1000);
      el.textContent = relTime(date, now);
      el.classList.toggle('is-warn', seconds >= 1800 && seconds < 7200);
      el.classList.toggle('is-danger', seconds >= 7200);
    });

    $$('[data-duration-s]').forEach(function (el) {
      var seconds = Number(el.getAttribute('data-duration-s'));
      el.textContent = isFinite(seconds) ? fmtDuration(seconds) : '—';
    });

    $$('[data-refresh-every]').forEach(function (el) {
      var every = parseInt(el.getAttribute('data-refresh-every'), 10);
      if (!isFinite(every) || every <= 0) return;
      var next = parseInt(el.getAttribute('data-next-refresh'), 10);
      if (!isFinite(next) || next <= 0) {
        next = now + every * 1000;
        el.setAttribute('data-next-refresh', String(next));
      }
      var left = Math.max(0, Math.ceil((next - now) / 1000));
      var counter = el.querySelector('[data-refresh-count]');
      if (counter) counter.textContent = String(left);
      if (now >= next) {
        el.setAttribute('data-next-refresh', String(now + every * 1000));
        scheduleRefresh();
      }
    });
  }

  window.setInterval(tick, 1000);

  document.addEventListener('visibilitychange', function () {
    if (!document.hidden) {
      stopFlash();
      if (!streamUrl) return;
      pollOnce();
      if (pollTimer) startPolling(pollIntervalMs);
    }
  });

  /* ------------------------------------------------------------------------
     9. 抽屉 / 弹窗 / 焦点陷阱
     ------------------------------------------------------------------------ */
  /* 告警弹窗也走这套：Esc 关闭、Tab 焦点陷阱、锁背景滚动全部复用，
     不必再写一份（见 §7）。 */
  function visibleRoots() {
    return $$('.fg-drawer-root, .fg-modal-root, .fg-alarmpop').filter(function (el) {
      return !el.hidden;
    });
  }

  function openRoot(root) {
    if (!root) return;
    root.hidden = false;
    document.body.style.overflow = 'hidden';
    var focusable = root.querySelector(
      'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    );
    if (focusable) focusable.focus();
  }

  function closeRoot(root) {
    if (!root) return;
    root.hidden = true;
    if (!visibleRoots().length) document.body.style.overflow = '';
    /* 告警弹窗：复位「当前展示」并接上排队中的下一条 */
    if (root.id === 'fg-alarmpop') onAlarmClosed();
    /* 新建账号后的一次性密码弹窗关闭时刷新列表，让新账号出现 */
    if ((root.id === 'fg-onetime-modal' || root.id === 'fg-ingest-modal')
        && root.getAttribute('data-needs-reload') === '1') {
      root.removeAttribute('data-needs-reload');
      window.setTimeout(function () {
        window.location.reload();
      }, 200);
    }
  }

  function openDrawer(selector) {
    var root = typeof selector === 'string' && selector.charAt(0) === '#'
      ? $(selector)
      : null;
    if (!root) root = $('#fg-drawer-root') || $('.fg-drawer-root');
    openRoot(root);
  }

  function closeAllDrawers() {
    $$('.fg-drawer-root').forEach(function (el) {
      el.hidden = true;
    });
    if (!visibleRoots().length) document.body.style.overflow = '';
  }

  function closeAllRoots() {
    $$('.fg-drawer-root, .fg-modal-root').forEach(function (el) {
      el.hidden = true;
    });
    document.body.style.overflow = '';
  }

  function openModal(selector) {
    var root = typeof selector === 'string' ? $(selector) : selector;
    if (!root) return;  /* 目标不存在就别炸；有的页面不渲染某些弹窗 */
    /* ：改密弹窗每次打开先清空 —— 护士站是半公共区域，
       上一个人的输入不能留给下一个人看见（表单自己标 data-reset-on-open）。 */
    var resetForm = root.querySelector('form[data-reset-on-open]');
    if (resetForm) {
      resetForm.reset();
      var errEl = resetForm.querySelector('[data-form-error]');
      if (errEl) { errEl.hidden = true; errEl.textContent = ''; }
    }
    openRoot(root);
  }

  function closeModal(root) {
    closeRoot(root);
  }

  FG.openDrawer = openDrawer;
  FG.closeDrawers = closeAllDrawers;
  FG.openModal = openModal;
  FG.closeModal = closeModal;

  /* innerHTML 换进来的 <script> 浏览器不会执行（HTML 规范如此）。
     抽屉里的判定策略面板靠一段内联脚本驱动「预设 → 各字段」，
     不重建就等于下拉框是死的 —— 换成新节点让它跑一次。 */
  function execScripts(container) {
    if (!container) return;
    $$('script', container).forEach(function (old) {
      var s = document.createElement('script');
      Array.prototype.forEach.call(old.attributes, function (attr) {
        s.setAttribute(attr.name, attr.value);
      });
      s.textContent = old.textContent;
      old.parentNode.replaceChild(s, old);
    });
  }

  function loadDetail(url) {
    var root = $('#fg-drawer-root') || $('.fg-drawer-root');
    if (!root) {
      window.location.assign(url);
      return;
    }
    var target = new URL(url, window.location.origin);
    target.searchParams.set('drawer', '1');
    fetch(target.toString(), {
      method: 'GET',
      credentials: 'same-origin',
      cache: 'no-store',
      headers: { 'Accept': 'text/html', 'X-Requested-With': 'fg-drawer' }
    }).then(function (res) {
      if (res.status === 401) {
        redirectLogin();
        return null;
      }
      if (!res.ok) throw new Error('HTTP ' + res.status);
      return res.text();
    }).then(function (html) {
      if (!html) return;
      var doc = new DOMParser().parseFromString(html, 'text/html');
      var freshBody = doc.querySelector('.fg-drawer__body');
      var currentBody = root.querySelector('.fg-drawer__body');
      if (!freshBody || !currentBody || !freshBody.firstElementChild) {
        throw new Error('drawer body missing');
      }
      currentBody.innerHTML = freshBody.innerHTML;
      /* 面板里若带内联脚本（判定策略的预设下拉），innerHTML 不会执行它 */
      execScripts(currentBody);
      var freshHead = doc.querySelector('.fg-drawer__head');
      var currentHead = root.querySelector('.fg-drawer__head');
      if (freshHead && currentHead) currentHead.innerHTML = freshHead.innerHTML;
      var freshFoot = doc.querySelector('.fg-drawer__foot');
      var currentFoot = root.querySelector('.fg-drawer__foot');
      if (currentFoot) currentFoot.innerHTML = freshFoot ? freshFoot.innerHTML : '';
      openRoot(root);
      if (!root.hasAttribute('data-drawer-no-push')) {
        window.history.pushState({ fgDrawer: url }, '', url);
      }
      scanDynamic();
    }).catch(function () {
      window.location.assign(url);
    });
  }

  function initOverlayEvents() {
    document.addEventListener('click', function (event) {
      var opener = event.target.closest('[data-drawer-open]');
      if (opener) {
        event.preventDefault();
        openDrawer(opener.getAttribute('data-drawer-open') || opener.getAttribute('href'));
        return;
      }

      var closer = event.target.closest('[data-drawer-close]');
      if (closer) {
        event.preventDefault();
        closeRoot(closer.closest('.fg-drawer-root'));
        return;
      }

      var modalClose = event.target.closest('[data-modal-close]');
      if (modalClose) {
        event.preventDefault();
        var closeRootEl = modalClose.closest('.fg-modal-root');
        if (closeRootEl && closeRootEl.hasAttribute('data-modal-static')) return;
        closeRoot(closeRootEl);
        return;
      }

      var modalOpen = event.target.closest('[data-modal-open]');
      if (modalOpen) {
        event.preventDefault();
        openModal(modalOpen.getAttribute('data-modal-open'));
        return;
      }

      var detailOpen = event.target.closest('[data-detail-open]');
      if (detailOpen) {
        event.preventDefault();
        loadDetail(detailOpen.getAttribute('data-detail-open'));
        return;
      }

      var row = event.target.closest('[data-detail-url]');
      if (row && !event.target.closest('a, button, input, select, textarea, label')) {
        event.preventDefault();
        loadDetail(row.getAttribute('data-detail-url'));
        return;
      }

      var navToggle = event.target.closest('[data-nav-toggle]');
      if (navToggle) {
        event.preventDefault();
        var shell = $('.fg-shell');
        if (shell) shell.classList.toggle('is-nav-open');
        return;
      }

      if (event.target.closest('[data-nav-collapse]')) {
        event.preventDefault();
        toggleNavCollapse();
        return;
      }

      var refreshTrigger = event.target.closest('[data-refresh]');
      if (refreshTrigger) {
        event.preventDefault();
        if (document.querySelector('[data-refresh-region]')) {
          refreshRegions();
        } else {
          window.location.reload();
        }
        return;
      }

      if (event.target.closest('.fg-nav__scrim')) {
        var openShell = $('.fg-shell');
        if (openShell) openShell.classList.remove('is-nav-open');
      }
    });

    document.addEventListener('keydown', function (event) {
      if (event.key === 'Enter' || event.key === ' ') {
        var row = event.target.closest('[data-detail-url]');
        if (row && event.target === row) {
          event.preventDefault();
          loadDetail(row.getAttribute('data-detail-url'));
          return;
        }
      }
      if (event.key === 'Escape') {
        var roots = visibleRoots();
        if (roots.length) {
          var topRoot = roots[roots.length - 1];
          event.preventDefault();
          /* 强制态弹窗（首登改密）Esc 也关不掉（ ）。同上：只拦这条路。 */
          if (topRoot.hasAttribute('data-modal-static')) return;
          closeRoot(topRoot);
        }
        return;
      }
      if (event.key !== 'Tab') return;
      var roots = visibleRoots();
      if (!roots.length) return;
      var root = roots[roots.length - 1];
      var focusables = $$(
        'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), ' +
        'textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
        root
      ).filter(function (el) {
        return el.offsetParent !== null;
      });
      if (!focusables.length) return;
      var first = focusables[0];
      var last = focusables[focusables.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    });

    window.addEventListener('popstate', function () {
      var root = $('#fg-drawer-root') || $('.fg-drawer-root');
      if (root) closeRoot(root);
    });
  }

  /* ------------------------------------------------------------------------
     10. 处置动作（接警 / 到场 / 处置 / 误报 / 重开）
     ------------------------------------------------------------------------ */
  /* 动作成功后按新状态说下一步，不按点了哪个按钮说：
     RESOLVE 从 ACKED 也能直接走，按按钮名会说出「完成处置后请到场」这种反话。 */
  var NEXT_STEP_TEXT = {
    ACKED: '已接警 · 下一步「到场」',
    ON_SCENE: '已到场 · 下一步「完成处置」',
    RESOLVED: '处置完成 · 已进入闭环',
    FALSE_ALARM: '已标记误报 · 已进入闭环',
    PENDING: '已重开 · 回到待接警'
  };
  var SEGMENT_AFTER_STATE = {
    ACKED: 'ACKED',
    ON_SCENE: 'ON_SCENE',
    RESOLVED: 'PENDING',
    FALSE_ALARM: 'PENDING'
  };

  function findEventUid(el) {
    if (!el) return null;
    var holder = el.closest('[data-event-uid]');
    return el.getAttribute('data-event-uid') ||
      (holder ? holder.getAttribute('data-event-uid') : null);
  }

  function findRevision(el) {
    if (!el) return null;
    var holder = el.closest('[data-revision]');
    return el.getAttribute('data-revision') ||
      (holder ? holder.getAttribute('data-revision') : null);
  }

  function postAction(uid, payload, revision, trigger) {
    if (!uid) return Promise.resolve();
    payload.client_request_id = uuid();
    if (revision !== null && revision !== undefined && revision !== '') {
      var parsed = Number(revision);
      if (isFinite(parsed)) payload.expected_revision = parsed;
    }
    if (trigger && trigger.classList) trigger.classList.add('is-loading');
    return fetch('/api/v1/events/' + encodeURIComponent(uid) + '/actions', {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
      body: JSON.stringify(payload)
    }).then(function (res) {
      if (res.status === 401) {
        redirectLogin();
        return null;
      }
      if (res.status === 409) {
        return res.json().catch(function () { return {}; }).then(function (data) {
          var message = (data.error && data.error.message) ||
            '该事件已被他人处理，请刷新后重试';
          toast(message, 'warn', 7000);
          scheduleRefresh();
          return null;
        });
      }
      if (!res.ok) {
        return res.json().catch(function () { return {}; }).then(function (data) {
          var message = (data.error && data.error.message) || '处置失败，请重试';
          toast(message, 'error', 7000);
          return null;
        });
      }
      return res.json().catch(function () { return {}; }).then(function (data) {
        var who = (data.assignee && data.assignee.display_name) ||
          body.getAttribute('data-user-name') || '';
        /* 动作做完要告诉人「下一步在哪」，而不是一句「已记录」。
           工作台把三步拆成了三段队列，不说清下一步，等于让人自己去猜。 */
        var msg = NEXT_STEP_TEXT[data && data.disposition_state] || '已记录';
        toast(msg + (who ? '，处置人：' + who : ''), 'ok', 4000);
        closeAllRoots();
        /* 接警/到场/闭环都会改变「还没人接」的条数 ⇒ 当场刷角标，别等 30 秒轮询。 */
        refreshPendingCount();
        /* 提交完的处置表单必须收起来。展开着的 form 会让 regionBusy 一直拦着
           工作区刷新 —— 动作条永远停在旧状态，看起来就是「点了没反应」。 */
        $$('[data-refresh-region="workspace"] form[data-action-form]').forEach(function (form) {
          form.reset();
          form.hidden = true;
          var err = form.querySelector('[data-form-error]');
          if (err) err.hidden = true;
        });
        /* 接警/到场后这条已经不在当前分段里了：直接跟到新分段并选中它。
           仅在工作台这么做——/alerts 的动作是从抽屉里发的，那里没有队列，
           跳 ?disposition= 会把人从告警中心拽走。 */
        var nextSeg = SEGMENT_AFTER_STATE[data && data.disposition_state];
        if (nextSeg && $('[data-refresh-region="workspace"]')) {
          /* 闭环回「待接警」时不能带 focus：服务端只要有 focus 就优先显示那条，
             哪怕它已经不属于这个分段——带过去就还是停在刚闭环的事件上。 */
          var jump = '?disposition=' + nextSeg;
          /* 点位筛选要跟着走（ ）：点了灯角的筛选钮之后接警，
             动作做完跳段时把 point 丢掉 = 护士眼前「筛着筛着变回全病区」。
              getAll 而不是 get：多选是重复参数 `?point=A&point=B`，
             get() 只会拿回第一个 —— 筛两个、跳完变一个，静默丢一半。 */
          new URL(window.location.href).searchParams.getAll('point').forEach(function (k) {
            if (k) jump += '&point=' + encodeURIComponent(k);
          });
          if (nextSeg !== 'PENDING') {
            jump += '&focus=' + encodeURIComponent(uid);
          }
          loadWorkspace(jump);
        } else {
          scheduleRefresh();
        }
      });
    }).catch(function () {
      toast('网络异常，请重试', 'error', 6000);
    }).then(function () {
      if (trigger && trigger.classList) trigger.classList.remove('is-loading');
    });
  }

  function collectActionPayload(form) {
    var payload = {};
    var outcome = form.querySelector('input[name="outcome"]:checked');
    if (outcome) payload.outcome = outcome.value;
    var reason = form.querySelector('input[name="reason"]:checked');
    if (!reason) {
      var reasonSelect = form.querySelector('select[name="reason"]');
      if (reasonSelect) reason = reasonSelect;
    }
    if (reason) payload.reason = reason.value;
    var note = form.querySelector('textarea[name="note"], input[name="note"]');
    if (note && note.value.trim()) payload.note = note.value.trim();
    return payload;
  }

  function revealActionPanel(action, trigger) {
    var scope = trigger.closest('.fg-drawer') || document;
    if (action === 'RESOLVE') {
      var form = scope.querySelector('form[data-action-form][data-action="RESOLVE"]');
      if (form) {
        form.hidden = false;
        if (form.scrollIntoView) form.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
        var first = form.querySelector('input, textarea, button');
        if (first) first.focus();
        return true;
      }
    }
    if (action === 'FALSE_ALARM') {
      var modal = $('#fg-false-modal');
      if (modal) {
        /* 弹窗里的 uid 是服务端烘进去的；触发按钮的 uid 才是「这一次要标的那条」。
           两者本该相同，但工作台换事件时容易留下旧弹窗（见 relocateModals 的注释）。
           以按钮为准回填，写法和 REOPEN 一致 —— 误报不可撤销，不能赌它们相同。 */
        var fuid = findEventUid(trigger);
        var frev = findRevision(trigger);
        var fform = modal.querySelector('form[data-action-form]');
        if (fform && fuid) {
          fform.setAttribute('data-event-uid', fuid);
          if (frev !== null && frev !== undefined) fform.setAttribute('data-revision', frev);
          var note = fform.querySelector('textarea');
          if (note && fuid) note.id = 'false-note-' + fuid;
        }
        openModal(modal);
        return true;
      }
    }
    if (action === 'REOPEN') {
      /* 台账整张表共用一个重开弹窗：uid 不能烘在 HTML 里，否则点第二行还是重开第一行。
         以触发按钮上的 data-event-uid / data-revision 为准回填。
         详情抽屉里那份自己带着 uid，回填成同一个值，无副作用。
         两个 id：台账页那份要避开抽屉注入的同名节点（见 _reopen_modal.html）。 */
      var reopen = $('#fg-reopen-modal') || $('#fg-reopen-modal-row');
      if (reopen) {
        var ruid = findEventUid(trigger);
        var rrev = findRevision(trigger);
        var rform = reopen.querySelector('form[data-action-form]');
        if (rform && ruid) {
          rform.setAttribute('data-event-uid', ruid);
          if (rrev !== null && rrev !== undefined) rform.setAttribute('data-revision', rrev);
        }
        openModal(reopen);
        return true;
      }
    }
    return false;
  }

  function initActions() {
    document.addEventListener('click', function (event) {
      var trigger = event.target.closest('button[data-action], a[data-action]');
      if (!trigger) return;
      if (trigger.closest('form[data-action-form]')) return;
      var action = trigger.getAttribute('data-action');
      if (!action) return;
      event.preventDefault();

      var uid = findEventUid(trigger);
      var revision = findRevision(trigger);
      if (!uid) return;

      if (action === 'RESOLVE' || action === 'FALSE_ALARM' || action === 'REOPEN') {
        if (revealActionPanel(action, trigger)) return;
      }
      postAction(uid, { action: action }, revision, trigger);
    });

    document.addEventListener('submit', function (event) {
      var form = event.target.closest('form[data-action-form]');
      if (!form) return;
      event.preventDefault();
      var uid = form.getAttribute('data-event-uid') || findEventUid(form);
      var action = form.getAttribute('data-action');
      var revision = form.getAttribute('data-revision');
      if (!uid || !action) return;
      var payload = collectActionPayload(form);
      payload.action = action;
      var submit = form.querySelector('button[type="submit"]') || form.querySelector('button');
      postAction(uid, payload, revision, submit);
    });
  }

  /* ------------------------------------------------------------------------
     11. 通用 JSON 表单 / 登录 / 账号管理
     ------------------------------------------------------------------------ */
  function formToObject(form) {
    var result = {};
    /* 逐个元素取，而不是 FormData：需要读 `data-nest` 把字段包进嵌套对象。
       带 data-nest="params" 的字段会变成 {params: {name: value, ...}} ——
       判定策略接口就是这种结构（扁平提交会被 extra=forbid 判 422）。 */
    var els = form.querySelectorAll('input[name], select[name], textarea[name]');
    Array.prototype.forEach.call(els, function (el) {
      var key = el.name;
      if (!key || key.charAt(0) === '_') return;
      if ((el.type === 'checkbox' || el.type === 'radio') && !el.checked) return;
      var value = el.value;
      /* 密码不 trim：首尾空格也可能是密码的一部分 */
      if (key.indexOf('password') < 0 && typeof value === 'string') {
        value = value.trim();
      }
      if (el.getAttribute('data-multi') === 'array') {
        if (!Array.isArray(result[key])) result[key] = [];
        result[key].push(value);
        return;
      }
      var nest = el.getAttribute('data-nest');
      if (nest) {
        if (!result[nest]) result[nest] = {};
        result[nest][key] = value;
      } else {
        result[key] = value;
      }
    });
    return result;
  }

  function showFormError(form, message) {
    var box = form.querySelector('[data-form-error]');
    if (box) {
      box.textContent = message;
      box.hidden = false;
    } else {
      toast(message, 'error', 6000);
    }
  }

  function submitLogin(form) {
    var button = form.querySelector('button[type="submit"]');
    var loginInput = form.querySelector('input[name="login_name"]');
    var passwordInput = form.querySelector('input[name="password"]');
    var loginName = loginInput ? loginInput.value.trim() : '';
    var password = passwordInput ? passwordInput.value : '';

    if (!loginName || !password) {
      showFormError(form, '请输入账号和密码');
      return Promise.resolve();
    }

    ensureAudio();
    if (button) {
      button.classList.add('is-loading');
      button.disabled = true;
    }

    return fetch('/api/v1/auth/login', {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
      body: JSON.stringify({ login_name: loginName, password: password })
    }).then(function (res) {
      return res.json().catch(function () { return {}; }).then(function (data) {
        if (res.ok) {
          var target = data.redirect_to || form.getAttribute('data-default-redirect') || '/';
          window.location.assign(target);
          return;
        }
        var message = (data.error && data.error.message) || '账号或密码不正确';
        showFormError(form, message);
      });
    }).catch(function () {
      showFormError(form, '网络异常，请稍后重试');
    }).then(function () {
      if (button) {
        button.classList.remove('is-loading');
        button.disabled = false;
      }
    });
  }

  function showOneTimePassword(data) {
    var modal = $('#fg-onetime-modal');
    var password = data && (data.temp_password || data.password || data.one_time_password);
    if (!modal) {
      if (password) toast('临时密码：' + password, 'ok', 15000);
      return;
    }
    var code = modal.querySelector('[data-onetime-code]');
    if (code) code.textContent = password || '—';
    var who = modal.querySelector('[data-onetime-user]');
    if (who) {
      who.textContent = (data.user && data.user.display_name) ||
        data.display_name || '';
    }
    modal.setAttribute('data-needs-reload', '1');
    openModal(modal);
  }

  /* 新建设备返回的 ingest key：明文只出现一次，必须当场展示并让人抄走。 */
  function showIngestKey(data) {
    var modal = $('#fg-ingest-modal');
    var key = (data && data.ingest_key) || '';
    if (!modal) {
      if (key) toast('接入密钥：' + key, 'ok', 30000);
      return;
    }
    var codeEl = modal.querySelector('[data-ingest-code]');
    if (codeEl) {
      codeEl.textContent = (data.device && data.device.device_code) || '';
    }
    var keyEl = modal.querySelector('[data-ingest-key]');
    if (keyEl) keyEl.textContent = key || '—';
    modal.setAttribute('data-needs-reload', '1');
    openModal(modal);
  }

  function submitApiForm(form) {
    var endpoint = form.getAttribute('data-api-endpoint') || form.getAttribute('action');
    var method = (form.getAttribute('data-api-method') || 'POST').toUpperCase();
    if (!endpoint) return Promise.resolve();

    var matchInput = form.querySelector('[data-match]');
    if (matchInput) {
      var source = $(matchInput.getAttribute('data-match'));
      if (source && matchInput.value !== source.value) {
        showFormError(form, '两次输入的密码不一致');
        return Promise.resolve();
      }
    }

    var button = form.querySelector('button[type="submit"]');
    if (button) {
      button.classList.add('is-loading');
      button.disabled = true;
    }
    return fetch(endpoint, {
      method: method,
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
      body: JSON.stringify(formToObject(form))
    }).then(function (res) {
      return res.json().catch(function () { return {}; }).then(function (data) {
        if (res.status === 401) {
          redirectLogin();
          return;
        }
        if (!res.ok) {
          var message = (data.error && data.error.message) || '操作失败，请重试';
          /* 422 带上具体字段：只显示「请求参数不合法」根本没法排查 */
          var fields = data.error && data.error.fields;
          if (fields && fields.length) {
            var f = fields[0];
            var loc = (f.loc || []).filter(function (x) { return x !== 'body'; }).join('.');
            if (loc) message += '（' + loc + '：' + (f.msg || '') + '）';
          }
          showFormError(form, message);
          return;
        }
        if (data.ingest_key) {
          showIngestKey(data);
          return;
        }
        if (data.temp_password || data.one_time_password) {
          showOneTimePassword(data);
          return;
        }
        /* 弹窗里的表单：成功后关窗、留在原地，不跳页（改密弹窗用， ）。
            关窗前先清空 —— 密码框留着明文，等于把刚设的密码摆在屏幕上。 */
        if (form.hasAttribute('data-close-modal')) {
          var modalRoot = form.closest('.fg-modal-root');
          form.reset();
          if (modalRoot) closeRoot(modalRoot);
        }
        var successUrl = form.getAttribute('data-success-url');
        var successText = form.getAttribute('data-success-text') || '已保存';
        toast(successText, 'ok', 3000);
        if (successUrl) {
          window.setTimeout(function () {
            window.location.assign(successUrl);
          }, 500);
        } else if (form.getAttribute('data-reload') !== '0') {
          window.setTimeout(function () {
            window.location.reload();
          }, 600);
        }
      });
    }).catch(function () {
      showFormError(form, '网络异常，请稍后重试');
    }).then(function () {
      if (button) {
        button.classList.remove('is-loading');
        button.disabled = false;
      }
    });
  }

  function callApi(url, method, payload, trigger) {
    if (!url) return Promise.resolve();
    if (trigger) {
      trigger.classList.add('is-loading');
      trigger.disabled = true;
    }
    return fetch(url, {
      method: method || 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
      body: JSON.stringify(payload || {})
    }).then(function (res) {
      return res.json().catch(function () { return {}; }).then(function (data) {
        if (res.status === 401) {
          redirectLogin();
          return;
        }
        if (!res.ok) {
          var message = (data.error && data.error.message) || '操作失败，请重试';
          toast(message, 'error', 6000);
          return;
        }
        if (data.ingest_key) {
          showIngestKey(data);
          return;
        }
        if (data.temp_password || data.one_time_password) {
          showOneTimePassword(data);
          return;
        }
        toast('已保存', 'ok', 3000);
        window.setTimeout(function () {
          window.location.reload();
        }, 600);
      });
    }).catch(function () {
      toast('网络异常，请重试', 'error', 6000);
    }).then(function () {
      if (trigger) {
        trigger.classList.remove('is-loading');
        trigger.disabled = false;
      }
    });
  }

  function initApiForms() {
    document.addEventListener('submit', function (event) {
      var form = event.target.closest('form[data-api-form], form[data-login-form]');
      if (!form) return;
      event.preventDefault();
      if (form.hasAttribute('data-login-form')) {
        submitLogin(form);
      } else {
        submitApiForm(form);
      }
    });

    document.addEventListener('click', function (event) {
      var trigger = event.target.closest('[data-api-action]');
      if (!trigger) return;
      event.preventDefault();
      var confirmText = trigger.getAttribute('data-confirm');
      if (confirmText && !window.confirm(confirmText)) return;
      var payload = {};
      var bodyText = trigger.getAttribute('data-api-body');
      if (bodyText) {
        try {
          payload = JSON.parse(bodyText);
        } catch (err) {
          payload = {};
        }
      }
      callApi(
        trigger.getAttribute('data-api-url'),
        trigger.getAttribute('data-api-method') || 'POST',
        payload,
        trigger
      );
    });
  }

  /* ------------------------------------------------------------------------
     12. 密码显示 / 复制 / 提示音开关
     ------------------------------------------------------------------------ */
  function initSmallWidgets() {
    document.addEventListener('click', function (event) {
      var toggle = event.target.closest('[data-toggle-password]');
      if (toggle) {
        event.preventDefault();
        var input = $(toggle.getAttribute('data-toggle-password'));
        if (!input) return;
        var show = input.type === 'password';
        input.type = show ? 'text' : 'password';
        toggle.setAttribute('aria-pressed', show ? 'true' : 'false');
        var label = toggle.querySelector('[data-eye-label]');
        if (label) label.textContent = show ? '隐藏' : '显示';
        return;
      }

      /* 演示账号一键填入：只填不提交，用户仍要自己点「登录」。
         不自动提交是刻意的——自动提交会让「登录」这个动作在用户没看见的情况下发生。 */
      var demo = event.target.closest('[data-demo-fill]');
      if (demo) {
        event.preventDefault();
        var nameInput = $('#login-name');
        var passInput = $('#login-password');
        if (nameInput) nameInput.value = demo.getAttribute('data-login') || '';
        if (passInput) {
          passInput.value = demo.getAttribute('data-password') || '';
          passInput.type = 'password';   // 还原为掩码状态
          /* 光把 type 改回 password 不够 —— 眼睛按钮自己记着"现在是显示态"
             （aria-pressed / 文案「隐藏」）。不一起复位，就会留下
             「字是黑的（掩码）但按钮写着『隐藏』」这种自相矛盾的状态，
             而且它只在「先点显示、再点演示账号」这一条路径上出现，
             正好是演示时最容易走到的路径。 */
          var eye = document.querySelector('[data-toggle-password="#login-password"]');
          if (eye) {
            eye.setAttribute('aria-pressed', 'false');
            var eyeLabel = eye.querySelector('[data-eye-label]');
            if (eyeLabel) eyeLabel.textContent = '显示';
          }
        }
        var submit = document.querySelector('.fg-login__submit');
        if (submit && submit.focus) submit.focus();
        return;
      }

      var copy = event.target.closest('[data-copy]');
      if (copy) {
        event.preventDefault();
        var target = $(copy.getAttribute('data-copy'));
        var text = target ? (target.textContent || '') : '';
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(text).then(function () {
            toast('已复制到剪贴板', 'ok', 2000);
          }).catch(function () {
            toast('复制失败，请手动抄写', 'warn', 4000);
          });
        } else {
          toast('请手动抄写', 'warn', 4000);
        }
      }
    });

    /* 提示音开关：顶栏用户菜单里一个，设置页里可能还有——按 [data-sound-toggle] 全绑上，
       几处 UI 共享同一个 localStorage 键，改一处其余几处同步。 */
    var soundToggles = $$('[data-sound-toggle]');
    soundToggles.forEach(function (box) {
      box.checked = soundEnabled();
      box.addEventListener('change', function () {
        try {
          window.localStorage.setItem(SOUND_KEY, box.checked ? '1' : '0');
        } catch (err) {
          /* 隐私模式下忽略 */
        }
        soundToggles.forEach(function (other) { other.checked = box.checked; });
        if (box.checked) {
          ensureAudio();
          beep();
          toast('提示音已开启', 'ok', 2000);
        } else {
          toast('提示音已关闭', 'ok', 2000);
        }
      });
    });

    /* 窄屏下头像也直接展开个人卡片（卡片已按视口宽度铺满）。 */

  }

  /* ------------------------------------------------------------------------
     13. 会话空闲 30 分钟自动登出（前端计时，后端仍校验）
     ------------------------------------------------------------------------ */
  function initIdleTimeout() {
    if (!body.hasAttribute('data-authed')) return;
    var limit = 30 * 60 * 1000;
    var timer = null;

    function reset() {
      if (timer) window.clearTimeout(timer);
      timer = window.setTimeout(function () {
        window.location.assign('/login?reason=idle');
      }, limit);
    }

    ['click', 'keydown', 'mousemove', 'touchstart', 'scroll'].forEach(function (name) {
      document.addEventListener(name, reset, { passive: true });
    });
    reset();
  }

  function syncNavCollapseButton() {
    var btn = $('[data-nav-collapse]');
    if (!btn) return;
    var collapsed = document.documentElement.classList.contains('fg-nav-collapsed');
    var label = collapsed ? '展开导航' : '收起导航';
    btn.setAttribute('aria-expanded', collapsed ? 'false' : 'true');
    btn.setAttribute('aria-label', label);
    btn.setAttribute('title', label);
    var text = btn.querySelector('.fg-nav__text');
    if (text) text.textContent = label;
  }

  function toggleNavCollapse() {
    var collapsed = document.documentElement.classList.toggle('fg-nav-collapsed');
    try {
      localStorage.setItem('fg-nav-collapsed', collapsed ? '1' : '0');
    } catch (e) {}
    syncNavCollapseButton();
  }

  function initForcedPasswordModal() {
    var root = $('#fg-pwd-modal[data-force-open]');
    if (root) openModal(root);
  }

  function init() {
    initBus();
    initAlarmPopup();
    initOverlayEvents();
    initActions();
    initWorkbench();
    initApiForms();
    initSmallWidgets();
    initForcedPasswordModal();
    initIdleTimeout();
    syncNavCollapseButton();
    scanDynamic();

    if (visibleRoots().length) document.body.style.overflow = 'hidden';

    if (body.hasAttribute('data-authed') && streamUrl) {
      connectStream();
      startPolling(5000);
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
