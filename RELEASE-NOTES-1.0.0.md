# TensorFold 1.0.0

TensorFold 1.0.0 ships the native Zig engine and HTTP server. Checkpoint loading, tokenization, chat templates and inference run in the native process, with no Python engine required at runtime.

## Supported models

The release includes these qualified model and hardware combinations:

| Model | Qualified platform | Serving scope |
| --- | --- | --- |
| Nemotron 3.5 Lightning | Apple Metal, M1 through M5 | Native text serving and its draft head |
| Nemotron 3.5 Lightning | NVIDIA GB10, CUDA | Text serving, greedy and sampled |
| Qwen3.8 Flash Next | Apple Metal, M5 Ultra | Greedy text serving from the checkpoint, without a Python kernel recording |
| GLM-5.3-Flash | Apple Metal, two M5 Ultra Macs | Paired text serving |
| Qwen3.5-2B | Apple Metal, M5 Max | Native text serving |

Hardware support is specific to the combinations above. Qwen3.8-27B, Bonsai, Gemma 4, Qwen3.6 and DeepSeek-V4 are being ported for 1.0.x. Qwen3.8-27B passes drafted/plain equality, but remains outside 1.0.0 because its paired served decode and cold-prefill results are slower than Python 0.6.6.

## Native runtime and server

Flash Next builds its native layouts from the checkpoint and uses embedded Metal sources. It no longer needs a Python recording before serving.

The server exposes OpenAI chat and completion routes, Anthropic Messages and token counting, tool calls, streaming responses, tokenization. Health, metrics and an optional dashboard expose server status. `/v1/decisions` checks requests the way 0.6.6 does; label scoring for each family follows in 1.0.x. API-key controls, prompt caching and request cancellation are implemented in the native server.

Metal engines support configurable idle keepalive through `--keep-warm`. A load-time probe selects a prebuilt packed-kernel library when macOS 26.3's runtime compiler rejects that kernel.

Release archives target macOS arm64 and Linux aarch64 for the GB10. They contain `bin/tensorfold-native`, runtime information and license notices. CUDA needs a compatible NVIDIA driver and the qualified kernel assets described in the deployment instructions; paired Metal serving needs its transport and peer configuration. See [README.md](README.md) for installation and [RUNBOOK.md](RUNBOOK.md) for deployment.

The shipped commands include `--version`, `--help`, `capabilities --json`, `serve MODEL`, and `pull`, `models` and `info` for checkpoints in the Hugging Face cache. Use `serve MODEL --help` and the capabilities output for supported flags and backends.

## Concurrency

Nemotron runs concurrent requests in shared rounds on Metal and CUDA, and every stream equals its solo run. On an M5 Max, eight concurrent requests reach 436.7 tok/s together, against 262.8 for one. Flash Next and GLM-5.3 answer one request at a time in 1.0.0; their shared rounds follow in 1.0.x.

## Context compaction

`--compact-at auto|FRACTION` turns on context compaction, which is off by default. A conversation that would overflow is compacted instead of refused. The server keeps the system prompt and the recent turns word for word, turns the older turns into a structured memory note, and updates that note at each later compaction. `--compact-keep` sets how much recent text stays, and `--compact-memory DIR` also keeps each note as a Markdown file. The same request gives the same compaction and the same reply.

## CUDA in 1.0.0

On the GB10, greedy and sampled replies equal the one-token reference, seeds and every sampling rule behave as in 0.6.6, concurrent streams equal their solo runs, and health and metrics report device memory. Decode runs 1.01x to 1.10x the Python 0.6.6 engine on the same Spark. Other NVIDIA GPUs and the other families stay on the Python line for now.

## Moving from Python 0.6

The Python 0.6.6 engine remains on the `python-0.6` branch and the `v0.6.6` tag. Its historical release entries remain in [CHANGELOG.md](CHANGELOG.md). The native release's model table and capabilities define its supported options; older Python CLI features and model backends are maintained on the Python line. A `pip install` of 1.0.0 stops with directions instead of replacing a working 0.6.6 install.

TensorFold is Apache-2.0. Earlier code retains the bundled MIT notice, and model weights retain their own licenses; see [LICENSE](LICENSE), [NOTICE](NOTICE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Contributors

Thank you to all 145 public contributors who have sent TensorFold a pull request, a measurement or a bug report. Their work shaped both the Python releases and the native engine:

[@0mao0](https://github.com/0mao0),
[@321sssrt-bit](https://github.com/321sssrt-bit),
[@aditya1503](https://github.com/aditya1503),
[@AdrianBinDC](https://github.com/AdrianBinDC),
[@akol1](https://github.com/akol1),
[@Alexbob0](https://github.com/Alexbob0),
[@anvilsong](https://github.com/anvilsong),
[@Arminova](https://github.com/Arminova),
[@b-ostrov](https://github.com/b-ostrov),
[@barelyworkingcode](https://github.com/barelyworkingcode),
[@Benjamin-Wegener](https://github.com/Benjamin-Wegener),
[@benthecarman](https://github.com/benthecarman),
[@benwilson](https://github.com/benwilson),
[@BHCC2025](https://github.com/BHCC2025),
[@Bizuayeu](https://github.com/Bizuayeu),
[@BlivionIaG](https://github.com/BlivionIaG),
[@BobClawblaw](https://github.com/BobClawblaw),
[@borodach23](https://github.com/borodach23),
[@Boscoeuk](https://github.com/Boscoeuk),
[@boxabirds](https://github.com/boxabirds),
[@brandonmmusic-max](https://github.com/brandonmmusic-max),
[@bunnyfu](https://github.com/bunnyfu),
[@CerebralCoding](https://github.com/CerebralCoding),
[@cesarswong](https://github.com/cesarswong),
[@chadhurley25075-png](https://github.com/chadhurley25075-png),
[@chaog992](https://github.com/chaog992),
[@Charlie-Louis](https://github.com/Charlie-Louis),
[@Chedrian07](https://github.com/Chedrian07),
[@chris247474](https://github.com/chris247474),
[@christrade215](https://github.com/christrade215),
[@crescit](https://github.com/crescit),
[@cshintov](https://github.com/cshintov),
[@cwschroeder](https://github.com/cwschroeder),
[@DakotaTexas](https://github.com/DakotaTexas),
[@danyo1399](https://github.com/danyo1399),
[@Deesha08](https://github.com/Deesha08),
[@Defilan](https://github.com/Defilan),
[@DevRico003](https://github.com/DevRico003),
[@di37](https://github.com/di37),
[@drowzeys](https://github.com/drowzeys),
[@ecohash-co](https://github.com/ecohash-co),
[@edurdias](https://github.com/edurdias),
[@eleqtrizit](https://github.com/eleqtrizit),
[@ericlsimplifi](https://github.com/ericlsimplifi),
[@EugeneClaw](https://github.com/EugeneClaw),
[@feni6](https://github.com/feni6),
[@gbgbgbg](https://github.com/gbgbgbg),
[@GDACONSULT](https://github.com/GDACONSULT),
[@gecobattya](https://github.com/gecobattya),
[@gilby](https://github.com/gilby),
[@Gogo6969](https://github.com/Gogo6969),
[@gprot42](https://github.com/gprot42),
[@GraithSecurity](https://github.com/GraithSecurity),
[@grantoverton](https://github.com/grantoverton),
[@grearjake-star](https://github.com/grearjake-star),
[@greatyingzi](https://github.com/greatyingzi),
[@harrisonfriia](https://github.com/harrisonfriia),
[@haxudev](https://github.com/haxudev),
[@heitke](https://github.com/heitke),
[@hichaiuse](https://github.com/hichaiuse),
[@ivanfioravanti](https://github.com/ivanfioravanti),
[@jasontitus](https://github.com/jasontitus),
[@jayleaton](https://github.com/jayleaton),
[@jeffpeng3](https://github.com/jeffpeng3),
[@jeidbugs404](https://github.com/jeidbugs404),
[@jetnet](https://github.com/jetnet),
[@jkuepker](https://github.com/jkuepker),
[@johnymoo](https://github.com/johnymoo),
[@JordiPosthumus](https://github.com/JordiPosthumus),
[@JRaxworthy](https://github.com/JRaxworthy),
[@jregan-beasley-smc](https://github.com/jregan-beasley-smc),
[@jschmied](https://github.com/jschmied),
[@juliankang4](https://github.com/juliankang4),
[@kingjamez](https://github.com/kingjamez),
[@kky42](https://github.com/kky42),
[@lcgutierrez](https://github.com/lcgutierrez),
[@LECYWZA](https://github.com/LECYWZA),
[@lijian1999](https://github.com/lijian1999),
[@liumorrisclaw](https://github.com/liumorrisclaw),
[@LXD-8](https://github.com/LXD-8),
[@m-naoki-m](https://github.com/m-naoki-m),
[@mapamalu](https://github.com/mapamalu),
[@mcclanahanaman](https://github.com/mcclanahanaman),
[@MESevenJourney](https://github.com/MESevenJourney),
[@mgoldwasser](https://github.com/mgoldwasser),
[@MiaAI-Lab](https://github.com/MiaAI-Lab),
[@mikolaj92](https://github.com/mikolaj92),
[@millaguie](https://github.com/millaguie),
[@Mirrdhyn](https://github.com/Mirrdhyn),
[@Moutonc](https://github.com/Moutonc),
[@MovieMaker93](https://github.com/MovieMaker93),
[@mrpmorris](https://github.com/mrpmorris),
[@MV10](https://github.com/MV10),
[@NeoAiLabs](https://github.com/NeoAiLabs),
[@Nipale-ai](https://github.com/Nipale-ai),
[@nood-co1](https://github.com/nood-co1),
[@nullburn](https://github.com/nullburn),
[@olexale](https://github.com/olexale),
[@omar16100](https://github.com/omar16100),
[@optimisme](https://github.com/optimisme),
[@outcastofmusic](https://github.com/outcastofmusic),
[@paragontasx](https://github.com/paragontasx),
[@peacockesq](https://github.com/peacockesq),
[@philip-pentatonic](https://github.com/philip-pentatonic),
[@plotarmordev](https://github.com/plotarmordev),
[@pmeenan](https://github.com/pmeenan),
[@pulseandthread](https://github.com/pulseandthread),
[@quigles1977](https://github.com/quigles1977),
[@rafafortes](https://github.com/rafafortes),
[@raymondkpwong](https://github.com/raymondkpwong),
[@robertpitt](https://github.com/robertpitt),
[@RoscoeTT](https://github.com/RoscoeTT),
[@salmanarshad321](https://github.com/salmanarshad321),
[@samwang0041-star](https://github.com/samwang0041-star),
[@sanjaibalajee](https://github.com/sanjaibalajee),
[@satindergrewal](https://github.com/satindergrewal),
[@scottleimroth](https://github.com/scottleimroth),
[@sethforprivacy](https://github.com/sethforprivacy),
[@sfxnz](https://github.com/sfxnz),
[@shantanugoel](https://github.com/shantanugoel),
[@simon-lin88](https://github.com/simon-lin88),
[@simonmd](https://github.com/simonmd),
[@spenchey](https://github.com/spenchey),
[@squarrier](https://github.com/squarrier),
[@ss-cong](https://github.com/ss-cong),
[@styles01](https://github.com/styles01),
[@SxMShaDoW](https://github.com/SxMShaDoW),
[@sxuff](https://github.com/sxuff),
[@taussoe](https://github.com/taussoe),
[@tfolkman](https://github.com/tfolkman),
[@ThinkOffApp](https://github.com/ThinkOffApp),
[@Thotheris](https://github.com/Thotheris),
[@tinyapps](https://github.com/tinyapps),
[@tolewis](https://github.com/tolewis),
[@tomByrer](https://github.com/tomByrer),
[@tonydehnke](https://github.com/tonydehnke),
[@tournierjc](https://github.com/tournierjc),
[@tpischke](https://github.com/tpischke),
[@urtho](https://github.com/urtho),
[@vcruz305](https://github.com/vcruz305),
[@vinicius-symetrix](https://github.com/vinicius-symetrix),
[@wojo](https://github.com/wojo),
[@xjqx2z](https://github.com/xjqx2z),
[@Yuepixel](https://github.com/Yuepixel),
[@YvesLaRose](https://github.com/YvesLaRose).
