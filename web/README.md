# 衡护 HENGUARD · 云端 Web

给值班护士和护士长用的病区跌倒事件工作台。板端上行推送事件；判定策略由板端主动
长轮询下发，云端不主动连板子。

视觉规范：品牌色 `#0E5A4E`，画布 `#EDF1EE`，导航 `#0B2B26` 宽 248 像素，
命令条 72 像素。

部署在一台 2C4G 的云服务器上，代码里用 `<SERVER_IP>` 指代。域名已经解析到该主机，
但备案未完成，公网访问会被云厂商的接入层拦到备案提示页，所以备案通过前一律用
`http://<SERVER_IP>/` 访问。在服务器上 `curl -H 'Host: <域名>' http://127.0.0.1/`
拿到 200 不能证明域名可用，那条请求不经过接入层，必须从外网打域名才算。

## 技术栈

FastAPI 加 Uvicorn，单 worker；PostgreSQL 16 加 psycopg3，没有 ORM、没有 Alembic，
建表直接跑 `db/schema.sql`；页面是 Jinja2 服务端渲染，没有前端构建链、没有 npm，
改完刷新就生效；静态资源是手写的 `tokens.css` / `app.css` / `app.js`，零依赖、
不走 CDN；口令用 argon2id 哈希，会话是签名 Cookie。

不做 ORM、不做前后端分离，是因为单病区、十人以内、离线可跑，构建链带来的复杂度
大于收益。

单 worker 是硬约束。实时推送用的是进程内的 `asyncio.Queue` 广播，多进程之间
收不到彼此的事件，页面会出现"有人收到有人收不到"。

## 目录

```
app/
  main.py          应用装配、页面路由、异常处理
  config.py        pydantic-settings，从环境变量或本地 env 文件读
  db.py            连接池与 query / query_one / execute
  security.py      argon2 哈希、签名 Cookie、登录限流
  deps.py          current_user / require_role / 区域过滤 / 审计
  metrics.py       五个 KPI 的 SQL 与时间工具，模板和接口共用
  routes/          五个 API 路由组：auth / events / ingest / stats / admin
  templates/       页面模板
db/
  schema.sql       表、索引、只追加审计触发器、事实视图
  seed.sql         病区、示例账号、示例设备
static/            tokens.css / app.css / app.js
scripts/           建号、JSON 回放压测、与板端 SQLite 对账
deploy/            systemd unit、nginx 配置、环境变量模板
```

## 本地跑起来

```bash
cd web
python -m venv .venv && .venv/Scripts/activate     # Linux 用 source .venv/bin/activate
pip install -r requirements.txt
# 需要一台 PostgreSQL 14 以上，先建角色和库，再灌 schema 和 seed
psql -U postgres -d fallguard -v ON_ERROR_STOP=1 -f db/schema.sql
psql -U postgres -d fallguard -v ON_ERROR_STOP=1 -f db/seed.sql
export DATABASE_URL='postgresql://fg_app:<密码>@127.0.0.1:5432/fallguard'
export SECRET_KEY='<64 位随机串>'
python -m scripts.create_user        # 建第一个护士长账号
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

打开 `http://127.0.0.1:8000/login`，首次登录会被强制跳到设置页改密。
不要加 `--workers 2`，理由见上面那条硬约束。

健康检查接口 `GET /healthz` 不需要登录，返回数据库状态、当前 SSE 连接数、
最近一条事件时间、今日事件数、剩余磁盘；数据库异常时 `status` 变 `degraded`
并且 `db` 字段带上原因。

## 域名与反代

nginx 站点配置在 `deploy/nginx.conf`，几个要点：

- 监听 80，`server_name` 里同时写域名和 IP，备案下来就能直接用。
- `/static/` 走 alias，nginx 以 `www-data` 运行，所以站点根目录必须是 751
  （others 只有 x，能穿越不能列目录），`static` 下 755/644。目录如果是 750，
  页面能打开但样式全丢，`/static/*.css` 返回 403。
- `/api/v1/stream` 必须单开一个 location，关掉代理缓冲、把读超时放到 3600 秒，
  否则 SSE 会被 nginx 缓冲住，页面看着像"连上了但永远不更新"。
- 上 TLS 之后要把 `X-Forwarded-Proto` 打开，并给会话 Cookie 补上 `Secure`。

四个安全响应头（`X-Content-Type-Options`、`X-Frame-Options`、
`Content-Security-Policy`、`Referrer-Policy`）也加在 nginx 上。加在 nginx 而不是
应用里，是因为静态资源由 nginx 直接出、应用只绑回环、502 也是 nginx 自己出的，
应用层中间件覆盖不全。注意 nginx 的 `add_header` 只在当前层级没有其它
`add_header` 时才继承，以后新加 location 不要再写 `add_header`。

## 板端接入

板端配置里只需要改一行，把 `push_url` 指向云端的接入地址，路径里带设备码和
接入密钥。密钥在建设备时生成，库里只存它的哈希。上行事件的幂等键是
"设备、板端序号、板端时间戳"三元组。撤回消息不走这条幂等键，而是按
"人号、窗口起点、模型版本"在十分钟内匹配原事件，匹配不上就标成孤立撤回。

## 运维

```bash
systemctl status <Web 服务>
journalctl -u <Web 服务> -f
systemctl restart <Web 服务>   # 只有改 .py 才需要，改模板不用，Jinja2 会重读
```

审计表只追加：表上挂了禁止更新和禁止删除的触发器，应用账号连 UPDATE/DELETE
权限都没有，想清日志只能把表删了重建，是故意做成"难"的。账号不能删只能禁用，
禁用之后旧 Cookie 立即失效，因为每次请求都会回库查用户。

## 取舍与边界

单 worker 加进程内队列：换来的是实时推送和页面渲染都很简单，不需要额外的中间件。
代价是不能水平扩展，接入 HTTPS 或多实例要改推送这一层。

服务端渲染加零依赖静态资源：换来的是没有前端构建链、改完刷新即生效。代价是交互
重一点的地方要手写 JS。

审计只追加、账号只能禁用：换来的是痕迹不可改写。代价是清日志要把表删了重建。

刻意不做的部分：不做 HLS 与实时视频（云端只显示设备状态卡，画面在板端直播页）、
不做床位管理和患者姓名（统一用"点位"与"人员编号"）、不做多病区切换（一个账号绑
一个病区）、不做 CSRF token（会话 Cookie 是 `SameSite=Lax`，上 TLS 后补双提交
token）、不做 WebSocket（SSE 单向下行够用，反代配置也更简单）。

目前只有 HTTP，域名备案完成后才上 TLS 与 Secure Cookie。告警没有超时自动关单，
未处置的会一直留着。
