(function () {
  var wall = document.querySelector('[data-camwall]');
  if (!wall) return;

  var grid = wall.querySelector('[data-camwall-grid]');
  var empty = wall.querySelector('[data-camwall-empty]');
  var notice = wall.querySelector('[data-camwall-notice]');
  var tpl = document.querySelector('[data-camcell-tpl]');
  if (!grid || !tpl) return;

  var MAX = 9;
  var points = {};        // id -> {id,name,stream,active,btn}
  var cells = {};         // id -> DOM 节点
  var order = [];         // 已选 id，顺序 = 加入顺序 = 宫格顺序

  wall.querySelectorAll('[data-point-id]').forEach(function (btn) {
    var id = btn.getAttribute('data-point-id');
    points[id] = {
      id: id,
      name: btn.getAttribute('data-point-name') || '',
      stream: btn.getAttribute('data-point-stream') || '',
      active: btn.getAttribute('data-point-active') === '1',
      btn: btn
    };
    btn.addEventListener('click', function () { toggle(id); });
  });

  function fromUrl() {
    var raw = new URLSearchParams(location.search).get('points') || '';
    return raw.split(',')
      .map(function (s) { return s.trim(); })
      .filter(function (id) { return id && points[id]; });
  }

  function syncUrl() {
    var q = new URLSearchParams(location.search);
    if (order.length) q.set('points', order.join(','));
    else q.delete('points');
    var qs = q.toString();
    history.replaceState(null, '', location.pathname + (qs ? '?' + qs : ''));
  }

  /* 「单独查看」要把当前选择一起带过去，单画面页的「返回画面墙」才能原样回来。
     不带的话，看完大画面回来 = 刚拼好的宫格被清空（ 用户实测反馈）。 */
  function soloHref(id) {
    var q = order.length
      ? '?points=' + order.map(encodeURIComponent).join(',')
      : '';
    return '/head/camera/' + encodeURIComponent(id) + q;
  }

  function makeCell(p) {
    var node = tpl.content.firstElementChild.cloneNode(true);
    node.setAttribute('data-camera-stream', p.stream);
    node.querySelector('.fg-camcell__name').textContent = p.name;
    node.querySelector('.fg-camcell__solo').setAttribute('href', soloHref(p.id));
    grid.appendChild(node);
    /* 交给播放器（它自带「已初始化」判重，重复调用安全） */
    if (typeof window.fgInitCamera === 'function') window.fgInitCamera(node);
    return node;
  }

  function updateNotice() {
    if (!notice) return;
    if (order.length > 1) {
      notice.hidden = false;
      notice.textContent =
        '已选 ' + order.length + ' 路：每路约 2.5 Mbps，出网实测 4.78 Mbps，' +
        '同时播放建议不超过 2 路。';
    } else {
      notice.hidden = true;
      notice.textContent = '';
    }
  }

  var flashTimer = null;
  function flash(msg) {
    if (!notice) return;
    notice.hidden = false;
    notice.textContent = msg;
    if (flashTimer) clearTimeout(flashTimer);
    flashTimer = setTimeout(function () { updateNotice(); }, 4000);
  }

  function render() {
    /* 1) 先收掉被移出的格子 */
    Object.keys(cells).forEach(function (id) {
      if (order.indexOf(id) >= 0) return;
      var node = cells[id];
      /* DOM 移走不会自动断流：必须显式 stop，否则云端仍按「有观众」送流 */
      if (typeof node.__fgStop === 'function') node.__fgStop();
      node.remove();
      delete cells[id];
    });

    /* 2) 新加入的格子追加到末尾（顺序 = 选择顺序，不重排已有节点） */
    order.forEach(function (id) {
      if (!cells[id]) cells[id] = makeCell(points[id]);
    });

    /* 2.5) 已有格子的「单独查看」链接也要跟着新选择走（makeCell 只覆盖新格子） */
    Object.keys(cells).forEach(function (id) {
      var a = cells[id].querySelector('.fg-camcell__solo');
      if (a) a.setAttribute('href', soloHref(id));
    });

    grid.setAttribute('data-count', String(order.length));
    if (empty) empty.hidden = order.length > 0;

    /* 3) 左列勾选态 */
    Object.keys(points).forEach(function (id) {
      var on = order.indexOf(id) >= 0;
      points[id].btn.classList.toggle('is-on', on);
      points[id].btn.setAttribute('aria-pressed', on ? 'true' : 'false');
    });

    updateNotice();
    syncUrl();
  }

  function toggle(id) {
    var i = order.indexOf(id);
    if (i >= 0) {
      order.splice(i, 1);
    } else {
      if (order.length >= MAX) {
        flash('最多同时看 9 路，先取消一个再选。');
        return;
      }
      order.push(id);
    }
    render();
  }

  /* 初始选择：URL 优先；没给就默认第一个配了摄像头的点位。
     不默认选「没配摄像头」的点 —— 进页面一片「未配置」没有意义。 */
  var initial = fromUrl();
  if (!initial.length) {
    var first = Object.keys(points).filter(function (id) { return points[id].stream; })[0];
    if (first) initial = [first];
  }
  order = initial;
  render();
})();
