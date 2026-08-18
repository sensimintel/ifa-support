# SAM3 使用路径地图（ifa-support）

> 用途：一眼看清 SAM3 在本仓被谁用、用在哪、动它会波及什么。
> 最后核对：2026-08-17（去冗余后：双目链、8061 实验台、/sam3 页、Gradio UI 均已下线；控制面识别日志观测已接上）。
> 改到 SAM3 相关链路请同步更新本文。

## 0. 服务本体

| 项 | 值 |
|---|---|
| 部署 | 5090 本机 systemd `sam3.service`，源码 `model/sam3/sam3_server.py`，装机脚本 `model/sam3/setup.sh` |
| 端点 | `http://127.0.0.1:8013`（消费端读 `SAM3_ENDPOINT`；app.py 的代码默认值是历史遗留的 8012，现网 `.env` 指到 8013） |
| 接口 | `/v1/segment`（无状态单图，可带 debug）、`/v1/track`（短窗口序列）、`/v1/stream/start` + `/v1/stream/frame`（服务端滚动窗口长记忆，obj_id 跨请求稳定） |
| 显存 | 实占 ~4.2G（`SAM3_MEM_FRACTION=0.28`，流式 `window` 控瞬时占用） |
| 存活检查 | `tools/health/check.sh`：systemd 单元 + 8013 端口监听 |

## 1. 消费方总表（去冗余后只剩两个）

| # | 消费方 | 调用方式 | 产出 / 去向 | 现网状态 |
|---|---|---|---|---|
| A | **识别触发线程的 SAM3 门控**（`_sam3_gate_dets`） | 每词一路并发，直接跑**当前设备的 RGB 彩色帧**，优先 `/v1/stream/frame`（带 debug 捕获），老 server 回退 `/v1/segment` | ①food/drink 归一化框 → 命中即带框送 VLM 识别 ②sam3tune 生产观测（`src="gate"`）→ 浅体验区控制面 | **在跑**（`/experience` 抽屉里直传开关设为「关」时；开=不调 SAM3） |
| A' | 8060 单目 RGB 链（`_maybe_sam3cloud` → `_sam3cloud_refresh`） | 每词一路并发，优先 `/v1/stream/frame`，老 server 回退 `/v1/track` | ①SAM3 点云映射 ②SAM3 高亮点云 ③液体框证据 ④sam3tune 生产观测 | **停用**（`.env` `DISABLE_MONO_PIPELINE=1`）。保留为「DA3+SAM3+点云+高亮」的完整能力样本，停用即零成本 |
| B | `/sam3tune` 调优页（`/api/sam3tune/*`） | `/v1/segment` 带 debug（presence/top-K） | 调参可视化 + **生产打分口径**落 `sam3_score_cfg.json`（词表/α/阈值），A 每帧读 | 在用（手动触发） |

已删除的历史消费方（2026-08-17 去冗余）：双目 IR 链（`_stereo_sam3_overlays`，整条双目 DA3 链一并删）、
`/sam3` 手动调试页（能力被 sam3tune 覆盖）、8061 感知链路实验台 `exp_app.py`（产线识别链路的完整复刻）。

## 2. A 链展开（单目 RGB 链，停用中）

`_da3_frame_processor` 每帧 DA3 推理后调 `_maybe_sam3cloud`（后台单任务，忙时跳过本帧），
`_sam3cloud_refresh` 里对 `_get_score_cfg()["words"]` 的每个词并发跑 SAM3，另补一路
`SAM3_TEXT`(默认 `food`) 专供染色。一轮结果同时喂四个下游：

1. **第三图 · SAM3 点云映射**：mask → 深度分辨率 → 逐 mask 画 3D AABB 框
   → `/api/sam3cloud/*` → `/panel` 第三格。
2. **第四图 · SAM3 高亮点云**：同一轮 overlays 的「无框纯高亮」版，样式读
   `/api/sam3hl/config` → `/api/sam3hl/*`。
3. **液体框证据**：`_sam3_dets`（TTL 6s，**只收 drink，food mask 明确排除**）
   → `_sam3_recent_drinks()` → SAM3 口径的识别触发（仅 `export_format=glb` 分支）。
4. **sam3tune 生产观测**：`_sam3tune_record_prod` 把生产轮的结果写回 `/sam3tune` 页展示。

切设备时（`_track_stream_device`）流式 session、`_sam3_dets`、产物槽一并重置。

> `/experience` 的背景来源已收敛为「设备深度图 + 设备点云」两条硬件深度链路，
> 不再消费 A 的任何产物；A 的产物只在 `/panel` 可见。

## 3. 识别触发口径（2026-08-17 起）

识别（Qwen3-VL 出卡片）有两种触发口径，**二选一**，闸门单点收口在
`recog_direct.should_trigger()`：

| 口径 | 触发条件 | 送 VLM 的图 | 节奏 |
|---|---|---|---|
| **直传（默认）** | 主链路定时取选中设备最新 **RGB 帧**，不看 SAM3 | 只有图1原图（+ 去重参考图） | `/api/recog/direct/config` 的 `interval_s`（默认 0.5s）× `concurrency`（默认 1=串行·最新优先） |
| **SAM3 门控（直传关掉时）** | 同一条触发线程按同样的间隔取 RGB 帧，先跑 SAM3（生产词表 + food 词），**认出食物/饮品才送 VLM** | 图1原图 + 图2带框图（+ 参考图） | 同上间隔；每轮多一次 SAM3（约 0.5~0.9s）。单目链启用时它的液体框缓存也走这条 |

两种口径**共用同一条触发线程与帧源**，区别只在"要不要 SAM3 先筛一道"——**两边都会出识别卡**。
开关在 `/experience` 右下「调节」抽屉 →「识别触发（主链路直传 VLM）」。

历史：触发原本挂在 A'（单目 glb 分支，证据只收 drink，食物从不触发）；`5bb5aec`(2026-08-13)
因 `DISABLE_MONO_PIPELINE=1` 停单目导致识别一起停，把触发搬到双目左目 IR；2026-08-17
去冗余删掉双目链后，SAM3 门控改为直接跑触发线程手里的 RGB 帧——不再依赖任何 DA3 链，
彩色图也比带散斑的 IR 灰度更适合认食物，food 与 drink 都算证据。

## 3.4 识别词的 label（决定命中算食物还是液体）

词表每个词都带 `label`（`food` / `drink`），它**不进 SAM3 请求**（查询词是 `word`），
只在本地决定命中的类别，一路影响三处：

1. `_sam3_gate_dets` 产出的 dets 类别位；
2. `_draw_boxes` 的框颜色（food 红 / drink 蓝）——图2 的图例正是「红框=疑似食物，蓝框=疑似液体/容器」；
3. prompt 里的软接地信息「检测器在当前画面的命中：食物框×N、液体框×M」。

写入侧（唯一入口是 superadmin 控制面）按类别分两组下发 `[{word,label}]`。
历史坑：控制面早期是一个不分类的 tags 输入，只发词面字符串，后端一律兜底
`drink`——`food` 这个词因此长期被标成 drink，每轮都在告诉模型「画面里没有食物」
（线上实测连续多轮 `n_food=0 n_drink=3`），食物框也被画成蓝色。现在后端的兜底顺序是：
**显式 label → 沿用该词现有 label → `SAM3_TEXT_DEFAULT` 判 food、其余 drink**。

## 3.5 识别观测日志（2026-08-17 起）——控制面看得见的过程态

浅体验区控制面（superadmin `/ifa-support/experience`，经 `/da3-api` 反代到 8060）两个页签：

**SAM3 侧**：`_sam3_gate_dets` 每轮写 `_sam3tune_record_prod(src="gate")` → `/api/sam3tune/state|history`；
原尺寸帧按 `SAM3TUNE_FULL_KEEP`(4) 条 + 各设备实时条留 ndarray 引用（**懒编码**：只有有人点开
`/api/sam3tune/image/{id}/{raw|seg}` 时才 imencode，常态零编码开销），滚出窗口后该端点 404。
去冗余前写观测的只有停用中的 A' 单目链，控制面 SAM3 区因此长期为空（实测 `state` 的 `live: {}`）。
细节：流式步进本就带 debug 捕获（server 端是前向 hook 读已有输出，**不多跑推理**），presence 与
top-K 原始分白拿；补跑的 food 词也进日志但标 `role="highlight"`、不计 `n_inst`（口径统计只认配置词）；
写历史按 `SAM3TUNE_HIST_MIN_GAP`(1.5s) 采样，实时区不受限；`/api/sam3tune/history` 支持 `device`/`limit`。

**VLM 侧**：每一轮识别（含失败轮）整轮留痕。

| 项 | 内容 |
|---|---|
| 一条日志 | 请求图（原图缩略 + 门控口径下的带框图）、候选参考图、prompt 原文、模型/endpoint/max_tokens/temperature、模型原始返回（截断 `VLMLOG_RAW_MAX`=20000 字符）、解析出的每一项、五道去重闸门的逐项判定（`outcome[].gate`/`action`/`card_id`） |
| 容量 | 内存环形 `VLMLOG_MAX`=30 条，不落盘，重启即空 |
| 接口 | `GET /api/recoglog/list?device=&limit=`（列表态剥掉 prompt/raw/参考图，只给长度与摘要）、`GET /api/recoglog/{id}`（全文，原图不内嵌）、`GET /api/recoglog/{id}/image/{orig\|boxed}`（**真正送进请求体的那张原尺寸图**，最近 `full_keep`=8 条）、`POST /api/recoglog/clear` |
| 代码 | 缓冲与投影在 `recog_log.py`（纯逻辑、可单测），图片编码在 `app.py` 的 `_thumb_uri` |

与识别卡片流的区别：卡片是「桌上现在有什么」的结果态，日志是过程态——漏识别 / 幻觉 / 错并
只能在过程态看出来，stdout 那份审计日志上了展台没人去 ssh 翻。

### 链路耗时分段（`timings`）

每条日志带一份分段，**端到端从「帧到达 8060」起算到「本轮结果落卡」**（不从触发起算，
否则帧在缓存里等触发线程那段被藏掉；拿不到帧时刻时 `total_ms` 留空而不是换个起点冒充）：

| 段 | 含义 | 归属 |
|---|---|---|
| `frame_age_ms` | 帧到达 8060 → 本轮任务入队 | 推帧节奏 + 触发间隔 |
| `decode_ms` | JPEG 解码成 ndarray | 本机 CPU |
| `gate_ms` | SAM3 门控（直传口径无此段） | 本机 GPU + 同机 8013 调用 |
| `wait_ms` | 入队 → worker 取走 | 并发不足才明显 |
| `encode_ms` | 两张原图 JPEG q95 + base64 + JSON 序列化 | 本机 CPU |
| `http_ms` | 发请求 → 收完响应 | **网络往返(隧道) + 服务端排队 + 推理 + 回传** |
| `parse_ms` | 解析输出 + guardrail | 本机 CPU |
| `post_ms` | 去重五闸 + 建卡/合并 | 本机 CPU |

`http_ms` 在这一层拆不开网络与推理。日志另附 `tunnel_rtt_ms`——最近一次隧道探测
（GET `8011/v1/models`，同一条 5090→Mac→IAP→GCP 路径、请求体极小、服务端不推理）的
往返，作为**网络基线**对照，差额基本是服务端排队与模型推理；`req_bytes` 给出本轮请求体
大小（图片 base64 占大头，直接决定上行传输时间）。基线只读探测缓存，绝不在 worker 里
补发请求——那会把被测对象自己搅乱。

## 4. 开关与配置速查

| 项 | 位置 | 作用 |
|---|---|---|
| `SAM3_ENDPOINT` | `.env` | SAM3 服务地址（现网 `http://127.0.0.1:8013`） |
| `SAM3_STREAM_WINDOW` | `.env`，默认 5 | 流式服务端滚动窗口帧数 |
| `SAM3_TEXT` | `.env`，默认 `food` | 系统补跑的 food 查询词。**判据是词面**：口径词表里已有这个词面就不补（早期按 label 判，现网 food/drink 都标 label=drink，会把 food 原样再跑一遍——白花一次 SAM3，命中时同一物体还会出两个框） |
| `DISABLE_MONO_PIPELINE` | `.env`，现网 `1` | 停单目链：A' 的四个下游全停（不影响 A 的 SAM3 门控与识别） |
| `sam3_score_cfg.json` | 仓根（gitignored） | 生产词表（**每个词带 label=food\|drink**）+ presence α + 检测阈值，控制面写、A 每帧读 |
| `sam3hl_preset.json` | 仓根（gitignored） | 高亮/点渲染样式，`/panel` 与 `/experience` 抽屉共写 |
| `recog_direct_cfg.json` | 仓根（gitignored） | 直传识别开关/间隔/并发 |
| `SAM3_OBS_LOG` | `.env`，默认开 | =0 关掉 SAM3 观测写回（控制面 SAM3 区随之空），嫌它占识别触发线程节拍时的应急开关 |
| `VLMLOG_MAX` / `VLMLOG_RAW_MAX` | `app.py` 常量 | 识别日志条数上限 / 单条原始返回截断长度 |
| `SAM3TUNE_HIST_MIN_GAP` | `app.py` 常量，1.5s | 生产 SAM3 观测写历史的采样间隔（实时区不受限） |
| `SAM3TUNE_FULL_KEEP` | `app.py` 常量，4 | 留原尺寸帧（可点开看大图）的观测条数 |
| `OBS_THUMB_W` / `OBS_THUMB_Q` | `.env`，默认 640 / 80 | 观测缩略图口径（送 VLM 的图不受影响，始终是设备原帧） |

> 现网提醒：SAM3 服务是识别「关」模式（SAM3 门控）的前置依赖，**不能停**——停了
> 门控每轮报错、识别不出卡。只有确认长期只用「开」（直传）时才谈得上停常驻，
> 且要同步改 `tools/health/check.sh` 的预期清单。
