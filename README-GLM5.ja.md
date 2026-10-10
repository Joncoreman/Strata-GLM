# Strata で GLM-5.3-Flash（320B-A18B）をゲーミング PC 1台で動かす

[English](README-GLM5.md) · **日本語**

[Strata](https://github.com/Niko1221/Strata)（v0.1.41・MIT）のフォークです。**GLM-5.3-Flash**（3,200億パラメータ・
活性 180億・unsloth の **UD-IQ4_XS** GGUF・157GB）を、GPU 1枚＋システム RAM＋NVMe の **PC 1台**で動かします。
Strata 本体は Qwen3.8-Flash-Next 用で、GLM 対応はこのフォークで 2026年10月に追加したものです。

データは PC の外に出ません。サーバーは OpenAI 互換と Anthropic 互換の API を話すので、Claude Code・Codex・opencode・
Cursor・各種チャット画面からそのまま使えます。

## 開発機での実測

RTX 5090（32GB）・Core Ultra 9 285K（24コア）・128GB DDR5-5200（2チャンネル）・NVMe・Windows 11・文脈 262K。
`tools/glm5_ref/glm_ab_speed.py` で、温めたサーバーに 256トークンを答えさせた値です。

| | 読み込み（prefill） | 生成（decode） |
|---|---:|---:|
| 短い会話（入力 86トークン） | 1.7秒 | **25 tok/s** |
| 560トークンの文書の後 | 5.5秒（100 tok/s） | 15.5 tok/s |
| 4,400トークンの文書の後 | 7.4秒（590 tok/s） | 13.3 tok/s |
| 32K トークンのプロンプト | 35〜45秒（700〜900 tok/s） | 約8 tok/s |

同じ PC・同じ日に llama.cpp（b11461・見つかった最良設定＝16スレッド・MoE 40層を CPU）で同じモデルを動かすと、
読み込み 3〜33 tok/s（86トークンで28秒・4,400トークンで135秒）、生成 8.1 tok/s でした。

上の「生成」は**専門家の省略**（後述）を使った値で、長い推論で100語に約1語が変わります。省略なしなら短い会話で
**20 tok/s**。読み込みの速さは省略に依存しません。

正しさ：読み込み経路と、省略以外の生成の改修はすべて、元の厳密な経路とビット単位で同じ出力です（改修のたびに
決定的な 4,400トークンの実行でトークン列を照合）。省略は「次の単語の確率分布の KL」で測り、要約で 0.016、
1,213トークンの推論文で 0.022（省略なし同士のばらつき＝床は 0.013）。

## 必要なもの

- Windows 10/11（Linux は GLM 経路では未検証・Strata 本体は対応）、Python 3.10 以上。
- NVIDIA GPU 24GB 以上（32GB で検証。24GB だと VRAM に載る専門家が減って遅くなります）。
  ビルドに Visual Studio 2022 Build Tools（C++）と CUDA toolkit 12.8 か 13。
- **RAM 128GB**（RAM 層に専門家 98GB を置きます。64GB だと大半が SSD からの読みになり生成が数倍遅くなります）。
- モデルを置く NVMe SSD（RAM に無い専門家は unbuffered で直接読みます）。GGUF 160GB＋パック 0.3GB。
- モデル：[unsloth/GLM-5.3-Flash-GGUF](https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF) の `UD-IQ4_XS`
  （5分割・157GB）。UD-IQ3_XXS（120GB）も生成は動きますが、読み込みは当面遅い経路になります（「分かっている制限」参照）。

## 入れ方

```bat
git clone <このリポジトリ> strata-glm5
cd strata-glm5
START-HERE.bat --setup          :: Strata 本体のセットアップ（Python・.venv・cmake/ninja/numpy/regex …）
build-glm5.bat                  :: この PC の GPU 向けにエンジンをビルド（RTX 50 は sm_120。RTX 30 は "build-glm5.bat 86"）
python tools\glm5_ref\make_glm5_config.py --gguf D:\models\UD-IQ4_XS\GLM-5.3-Flash-UD-IQ4_XS-00001-of-00005.gguf
START-GLM5.bat                  :: http://127.0.0.1:8097/v1 （RAM 層の固定に約1分）
```

`make_glm5_config.py` は、パック（`tools/iq_pack.py`＝密なテンソルと専門家の目次・0.3GB）をリポジトリの隣の
`Strata-data/packs/glm53_iq4xs` に作り、学習済みの専門家プロファイルを横に置き、この PC 向けの `strata-glm5.json`
（RAM 層＝総 RAM−30GB・スレッド＝物理コア−4）を書きます。他の設定はそのファイルを編集してください
（`strata-glm5.example.json` に説明があります）。`--skip` を付けると専門家の省略（生成 +30%）が入ります。

Strata 本体の `START-HERE.bat --setup` は Qwen のモデルも入れようとします。.venv ができた時点で止めても（Ctrl-C）、
最後まで通しても構いません。

### 使い方

クライアントの接続先を `http://127.0.0.1:8097/v1`（OpenAI：`/v1/chat/completions`、Anthropic：`/v1/messages`）、
モデル名 `glm-5.3-flash-strata` にします。Claude Code／opencode／Cursor の設定は Strata の
[docs/CLIENTS.md](docs/CLIENTS.md) と同じです。

## なぜ速いか（効かなかったことも）

設計と全測定は `glm-notes/計画_読み込みの作り替え_2026-10-09.md` にあります。要点だけ：

**読み込み（prefill）。** 専門家を「層ごとに歩く」：各層の専門家を 4K トークンの区切りごとでなく、最大 16K トークン
のまとまりにつき1回だけ VRAM へ運び、int8 のテンソルコア（llama.cpp の MMQ）で計算します。共有専門家・密な層・
注意側の射影もテンソルコアの GEMM、ルーターは cuBLAS 1回、長文の MLA 注意は WMMA で 32ヘッド/ブロック、SSD 層は
1層先に先読み、ホスト側の状態は4本の連続配列でルーターと並行に移します。4,400トークンが 59.8秒 → 7.4秒。

**生成（decode）。** RAM 層は固定して学習済みプロファイルの順に並べ、最頻の専門家は VRAM に置き、両層とも会話に
追従します（プロンプトの経路回数で初期化）。CPU の計算は1トークンでも AVX-VNNI。層の SSD 読みは読み専用スレッドで
RAM 側の計算と並行、RAM 層への昇格の読みは1窓先に。MLA の 64ヘッドの吸収積と、同じ入力を取る KDA の射影は1回の
起動に。会話の生成は 13.2 → 20.2 tok/s（答え不変）、省略ありで 25。

**専門家の省略**（`STRATA_GLM5_SKIP_KEEP=2 STRATA_GLM5_SKIP_MINW=0.10`）：そのトークンへの経路の重みが全体の 10%
未満で、VRAM に無い専門家は計算しない（上位2つは必ず計算）。[2×3090 レシピ](https://github.com/saitakarcesme/glm5.3-flash-2x3090)
の cache-aware routing に倣ったもの。既定はオフ。

**測って採らなかったもの**（すべてこの PC）：RTX 3060 の2台目 PC を層の段にする（頭が速くなった後は 14.4 対 20.2
＝3060 の1層は浮く時間の3倍かかる）、PCIe 経由で GPU にも専門家を計算させる（`--pcie-frac`・DMA と CPU が同じ RAM
帯域を取り合う）、MTP の下書き（窓を広げると CPU の仕事が比例して増える）、次の層の経路予測による SSD 先読み
（外れた読みの方が高い）、省略した重みの配り直し、少数専門家での自己投機（下書きが全部当たっても遅い）、GGUF の
マップ外しで unbuffered 読みを速くする（ここでは 15%・12MB の読み1本で NVMe の上限に届く）。

**今の時間の内訳**（短い会話・1トークン 50ms）：CPU の RAM 専門家 19〜25ms（帯域律速：プールは 44GB/s、この PC の
上限は 60GB/s。コアを速くしても効かず、効くのはメモリのチャンネル数）、GPU の約2,000 回の小さなカーネル起動 13ms、
SSD 3ms、残りは入れ替えと合図。

## 分かっている制限

- このフォークの GLM 経路は NVIDIA 1枚のみ（2台の段モードはコードにあるが現状は遅い。AMD は未検証）。
- UD-IQ3_XXS：生成は動く（IQ2_S/IQ3_S の専門家は CPU・GPU とも対応）が、注意側の重みが Q6_K のため読み込みは
  `STRATA_GLM5_PREFILL=0`（検証窓の経路）が要る。
- パックは量子化ごと。`make_glm5_config.py` は `glm53_<量子化名>` と名付ける。
- `tools/glm5_ref/` は開発用の台本（速度 A/B・精度の物差し `car_follow.py`／`car_kl.py`・Nsight 用）、`mn-test/`
  は2台実験。パスは `strata-glm5.json` から読む。

## 謝辞とライセンス

[Strata](https://github.com/Niko1221/Strata)（Niko1221 と Strata の貢献者・MIT）。このフォークはそのライセンスと
著作権表示（`LICENSE`）をそのまま引き継ぎます。GLM-5.3-Flash は Z.ai、GGUF は unsloth。llama.cpp の MMQ カーネルと
量子化形式（MIT）。比較・参考にしたもの：[Project Maya](https://github.com/mw00/project-maya)、
[glm-flash-lite](https://github.com/sybil-solutions/glm-flash-lite)、上記の 2×3090 レシピ。GLM 対応と高速化は
Claude（Anthropic）をペアプログラマとして書きました。上の数字はすべて開発機での実測で、`tools/glm5_ref/` の台本で
再現できます。
