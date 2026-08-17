# SAM3 使用路径地图（ifa-support）

> 用途：一眼看清 SAM3 在本仓被谁用、用在哪、动它会波及什么。
> 最后核对：2026-08-17（`c831ea5` + 直传识别 + 控制面识别日志观测）。改到 SAM3 相关链路请同步更新本文。

## 0. 服务本体

| 项 | 值 |
|---|---|
| 部署 | 5090 本机 systemd `sam3.service`，源码 `model/sam3/sam3_server.py`，装机脚本 `model/sam3/setup.sh` |
| 端点 | `http://127.0.0.1:8013`（消费端读 `SAM3_ENDPOINT`；app.py 的代码默认值是历史遗留的 8012，现网 `.env` 指到 8013） |
| 接口 | `/v1/segment`（无状态单图）、`/v1/track`（短窗口序列）、`/v1/stream/start` + `/v1/stream/frame`（服务端滚动窗口长记忆，obj_id 跨请求稳定） |
| 显存 | 实占 ~4.2G（`SAM3_MEM_FRACTION=0.28`，流式 `window` 控瞬时占用） |
| 存活检查 | `tools/health/check.sh`：systemd 单元 + 8013 端口监听 |
| 历史 | GCP gpu-g4-01 上的 legacy sam3(8001) 已停用腾显存，恢复方式见 odyss-models 该单元 README |

## 1. 消费方总表

| # | 消费方 | 调用方式 | 产出 / 去向 | 现网状态 |
|---|---|---|---|---|
| A | 8060 单目 RGB 链（`_maybe_sam3cloud` → `_sam3cloud_refresh`） | 每词一路并发，优先 `/v1/stream/frame`，老 server 回退 `/v1/track` | ①SAM3 点云映射 ②SAM3 高亮点云 ③液体框证据 ④sam3tune 生产观测 | **停用**（`.env` `DISABLE_MONO_PIPELINE=1`） |
| B | 8060 双目 IR 链（`_stereo_sam3_overlays`） | 左目 IR 每词 `/v1/stream/frame`（**独立 session 表**，与 A 的记忆完全隔离），带 debug 捕获 | ①双目点云染色 ②双目高亮点云 ③识别触发证据 ④sam3tune 生产观测 | **在跑**（g335 推 aux 时） |
| C | `/sam3` 调试页（`POST /api/sam3/run`） | 手动一次 `/v1/segment` + `/v1/track`，多词并发 | 页面上三张图（原图/分割/跟踪） | 可用，但取帧靠 A 的 `_recent_frames` —— A 停用时无帧可跑 |
| D | `/sam3tune` 调优页（`/api/sam3tune/*`） | `/v1/segment` 带 debug（presence/top-K） | 调参可视化 + **生产打分口径**落 `sam3_score_cfg.json`（词表/alpha/thresh），A 与 B 每帧读 | 在用（口径是 A/B 的输入） |
| E | `exp_app.py` 感知链路实验台（8061） | 无状态 `/v1/segment`，自带查询词/阈值/口径 | 实验台自己的标注帧 + VLM 卡片流 | 手动起停，与产线完全隔离 |

## 2. 逐条展开

### A. 单目 RGB 链（app.py，`DISABLE_MONO_PIPELINE=1` 时整条不注册）

`_da3_frame_processor` 每帧 DA3 推理后调 `_maybe_sam3cloud`（后台单任务，忙时跳过本帧），
`_sam3cloud_refresh` 里对 `_get_score_cfg()["words"]` 的每个词并发跑 SAM3，另补一路
`SAM3_TEXT`(默认 `food`) 专供染色。一轮结果同时喂四个下游：

1. **第三图 · SAM3 点云映射**：mask → 深度分辨率 → 逐 mask 画 3D AABB 框
   → `/api/sam3cloud/*` → `/panel` 第三格 + `/experience` 背景来源「SAM3点云」(`s3`)。
2. **第四图 · SAM3 高亮点云**：同一轮 overlays 的「无框纯高亮」版，样式读
   `/api/sam3hl/config` → `/api/sam3hl/*` → `/experience` 背景来源「高亮点云」(`hl`)。
3. **液体框证据**：`_sam3_dets`（TTL 6s，**只收 drink，food mask 明确排除**）
   → `_sam3_recent_drinks()` → 识别触发（仅 `export_format=glb` 分支）。
4. **sam3tune 生产观测**：`_sam3tune_record_prod` 把生产轮的结果写回 `/sam3tune` 页展示。

切设备时（`_track_stream_device`）流式 session、`_sam3_dets`、两路产物槽一并重置。

### B. 双目 IR 链（app.py）

`_da3_stereo_processor` 处理左右 IR 后，`_stereo_sam3_overlays` 对**左目 processed_image**
逐词流式步进（`_sam3_ir_stream_frame`，`_sam3_ir_sessions` 独立 session 表）。同样补一路
food 查询。下游：

1. **双目点云染色**：overlays 进 `build_stereo_pointcloud_glb` → `/panel` 第一格。
2. **双目高亮点云**：同 pred/overlays 追加构建，样式读同一份 `/api/sam3hl/config`
   → `/api/stereohl/status` → `/experience` 背景来源「双目高亮点云」(`stereo`)。
3. **识别触发**：overlays（**food + drink 都算**）→ 归一化框 + 该设备最新 RGB 帧
   → `_maybe_recognize`。`pred` 不传（IR 系深度与 RGB 不对齐，点云缩略图降级为无）。
4. **sam3tune 生产观测**（2026-08-17 起）：每轮结果写 `_sam3tune_record_prod(src="stereo")`
   → `/api/sam3tune/state|history` → 浅体验区控制面。A 停用后这是控制面唯一的 SAM3
   数据源（此前控制面 SAM3 区一直是空的）。补跑的 food 高亮词也进日志，标
   `role="highlight"` 且不计 `n_inst`——口径统计只认配置词。写历史按
   `SAM3TUNE_HIST_MIN_GAP`（1.5s）采样，实时区不受限。

### C/D. 调试与调优页

- `/sam3`：`POST /api/sam3/run` 手动跑，多词逗号分隔，取 `_recent_frames`（A 填的）。
- `/sam3tune`：presence/top-K debug（`pred_logits` 已是联合分，`forward_grounding` 钩子
  拿 NMS 前原始分），并写生产口径 `sam3_score_cfg.json` —— **这是 A/B 的词表来源**，
  改这里等于改生产链路的检测词与打分。

### E. 实验台 8061

`exp_app.py` 从 8060 只读接口拉帧 → 无状态 `/v1/segment`（自己的查询词/阈值/α）→
标注帧 + 完整 VLM 卡片流。不碰产线流式 session，可随便魔改。

## 3. 识别触发口径（2026-08-17 起）

识别（Qwen3-VL 出卡片）有两种触发口径，**二选一**，闸门单点收口在
`recog_direct.should_trigger()`：

| 口径 | 触发条件 | 送 VLM 的图 | 节奏 |
|---|---|---|---|
| **直传（默认）** | 主链路定时取选中设备最新 **RGB 帧**，不看 SAM3 | 只有图1原图（+ 去重参考图） | `/api/recog/direct/config` 的 `interval_s`（默认 0.5s）× `concurrency`（默认 1=串行·最新优先） |
| SAM3（直传关掉时） | A 的液体框缓存 或 B 的左目命中 | 图1原图 + 图2带框图（+ 参考图） | `RECOG_MIN_INTERVAL` 节流 + worker 丢积压 |

直传开启时 A/B 两处 SAM3 触发**整体让位**（代码保留，闸掉即可回退），SAM3 只继续供
背景可视化。开关在 `/experience` 右下「调节」抽屉 →「识别触发（主链路直传 VLM）」。

历史：触发原本只在 A（单目 glb 分支）；`5bb5aec`(2026-08-13) 因 `DISABLE_MONO_PIPELINE=1`
停单目导致识别一起停，把触发搬到 B。两处并存的表象来自这次搬家，不是双路并跑。

## 3.5 识别观测日志（2026-08-17 起）

每一轮 VLM 识别（含失败轮）整轮留痕，供浅体验区控制面可视化排障。缓冲与投影在
`recog_log.py`（纯逻辑，可单测），图片编码在 `app.py` 的 `_thumb_uri`。

| 项 | 内容 |
|---|---|
| 一条日志 | 请求图（原图缩略 + SAM3 口径下的带框图）、候选参考图、prompt 原文、模型/endpoint/max_tokens/temperature、模型原始返回（截断 `VLMLOG_RAW_MAX`=20000 字符）、解析出的每一项、五道闸门逐项判定（`outcome[].gate`/`action`/`card_id`） |
| 容量 | 内存环形 `VLMLOG_MAX`=30 条，不落盘，重启即空 |
| 接口 | `GET /api/recoglog/list?device=&limit=`（列表态剥掉 prompt/raw/参考图，只给长度与摘要）、`GET /api/recoglog/{id}`（全文）、`POST /api/recoglog/clear` |
| 消费方 | superadmin `/ifa-support/experience` 控制面「VLM 识别日志」页签 |

与卡片流的区别：卡片是「桌上现在有什么」的结果态，日志是过程态——漏识别 / 幻觉 /
错并只能在过程态看出来，stdout 那份审计日志上了展台没人去 ssh 翻。

## 4. 开关与配置速查

| 项 | 位置 | 作用 |
|---|---|---|
| `SAM3_ENDPOINT` | `.env` | SAM3 服务地址（现网 `http://127.0.0.1:8013`） |
| `SAM3_STREAM_WINDOW` | `.env`，默认 5 | 流式服务端滚动窗口帧数 |
| `SAM3_TEXT` | `.env`，默认 `food` | 补给染色的 food 查询词 |
| `DISABLE_MONO_PIPELINE` | `.env`，现网 `1` | 停单目链：A 的四个下游全停，算力让给双目 |
| `sam3_score_cfg.json` | 仓根（gitignored） | 生产词表 + presence α + 检测阈值，`/sam3tune` 写、A/B 每帧读 |
| `sam3hl_preset.json` | 仓根（gitignored） | 高亮/点云样式，`/panel` 与 `/experience` 抽屉共写 |
| `recog_direct_cfg.json` | 仓根（gitignored） | 直传识别开关/间隔/并发 |
| `VLMLOG_MAX` / `VLMLOG_RAW_MAX` | `app.py` 常量 | 识别日志条数上限 / 单条原始返回截断长度 |
| `SAM3TUNE_HIST_MIN_GAP` | `app.py` 常量，1.5s | 生产 SAM3 观测写历史的采样间隔（实时区不受限） |
| `SAM3_OBS_LOG` | `.env`，默认开 | =0 关掉 SAM3 观测写回（控制面 SAM3 区随之空），展台嫌它占双目构建线程时的应急开关 |
