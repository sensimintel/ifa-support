# SAM3 使用路径地图（ifa-support）

> 用途：一眼看清 SAM3 在本仓被谁用、用在哪、动它会波及什么。
> 最后核对：2026-08-17（去冗余后：双目链、8061 实验台、/sam3 页、Gradio UI 均已下线）。
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
| A | 8060 单目 RGB 链（`_maybe_sam3cloud` → `_sam3cloud_refresh`） | 每词一路并发，优先 `/v1/stream/frame`，老 server 回退 `/v1/track` | ①SAM3 点云映射 ②SAM3 高亮点云 ③液体框证据(SAM3 口径的识别触发) ④sam3tune 生产观测 | **停用**（`.env` `DISABLE_MONO_PIPELINE=1`）。保留为「DA3+SAM3+点云+高亮+触发识别」的唯一完整能力样本，停用即零成本 |
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
| SAM3（直传关掉时） | A 的液体框缓存命中 | 图1原图 + 图2带框图（+ 参考图） | `RECOG_MIN_INTERVAL` 节流 + worker 丢积压 |

直传开启时 SAM3 口径**整体让位**（代码保留，闸掉即可回退）。开关在 `/experience`
右下「调节」抽屉 →「识别触发（主链路直传 VLM）」。

历史：触发原本只在 A（单目 glb 分支）；`5bb5aec`(2026-08-13) 因 `DISABLE_MONO_PIPELINE=1`
停单目导致识别一起停，把触发搬到双目链；2026-08-17 改为直传口径，双目链随之删除。

## 4. 开关与配置速查

| 项 | 位置 | 作用 |
|---|---|---|
| `SAM3_ENDPOINT` | `.env` | SAM3 服务地址（现网 `http://127.0.0.1:8013`） |
| `SAM3_STREAM_WINDOW` | `.env`，默认 5 | 流式服务端滚动窗口帧数 |
| `SAM3_TEXT` | `.env`，默认 `food` | 补给染色的 food 查询词 |
| `DISABLE_MONO_PIPELINE` | `.env`，现网 `1` | 停单目链：A 的四个下游全停（当前唯一的 SAM3 生产消费方，因此现网 SAM3 只被 sam3tune 手动调用） |
| `sam3_score_cfg.json` | 仓根（gitignored） | 生产词表 + presence α + 检测阈值，`/sam3tune` 写、A 每帧读 |
| `sam3hl_preset.json` | 仓根（gitignored） | 高亮/点渲染样式，`/panel` 与 `/experience` 抽屉共写 |
| `recog_direct_cfg.json` | 仓根（gitignored） | 直传识别开关/间隔/并发 |

> 现网提醒：A 停用 + 双目链删除后，SAM3 服务在常规运行中已无自动调用者（只剩
> `/sam3tune` 手动触发）。要腾 5090 显存可以停 `sam3.service`，但**必须同步改
> `tools/health/check.sh` 的预期清单**，否则体检会把它误报成故障。
