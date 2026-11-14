<p align="center">
  <a href="https://ko-fi.com/ashxhart">Help Support TensorFold</a> ·
  <a href="https://x.com/tensorfold_dev">@tensorfold_dev on X</a>
</p>

<p align="center">
  <img src="assets/tensorfold-logo.png" alt="TensorFold logo" width="160">
</p>

# TensorFold

TensorFold serves language models from a Zig binary on Apple Silicon and NVIDIA GPUs.
The engine reads checkpoints, tokenizes requests and runs Metal or CUDA kernels directly.
Serving needs no Python or MLX installation.

Building with a coding agent? Have it load the [contributor skill](docs/recipes/SKILL.md) and follow [CONTRIBUTING.md](CONTRIBUTING.md#contributing-with-agents).

## Quick start

Start with Qwen3.5-2B, a small 4-bit checkpoint. The commands below download its pinned revision, then serve it as
`local-model`. Model weights have their own licenses.

### Mac with Apple Silicon

Install with Homebrew, then start the server:

```sh
brew install ashhart/tensorfold/tensorfold
tensorfold pull mlx-community/Qwen3.5-2B-MLX-4bit@93760be4f1f69842a46bc13dbdc0f19e291392a3
tensorfold serve mlx-community/Qwen3.5-2B-MLX-4bit --name local-model --context 8192 --no-thinking
```

### Linux with an NVIDIA GPU

Needs glibc 2.28 or newer and NVIDIA driver R580 or newer (CUDA 13.0). The [run guide](docs/run-guide.md#platform-setup)
lists the archive's requirements; Windows has [its own guide](docs/WINDOWS.md).

```sh
V=1.2.8
case "$(uname -m)" in
  aarch64|arm64) PLATFORM=linux-aarch64 ;;
  x86_64) PLATFORM=linux-x86_64 ;;
  *) echo 'No release archive for this CPU' >&2; exit 1 ;;
esac
curl -fLO "https://github.com/ashhart/TensorFold/releases/download/v$V/tensorfold-$V-$PLATFORM.tar.gz"
curl -fLO "https://github.com/ashhart/TensorFold/releases/download/v$V/tensorfold-$V-$PLATFORM.tar.gz.sha256"
sha256sum -c "tensorfold-$V-$PLATFORM.tar.gz.sha256"
tar -xzf "tensorfold-$V-$PLATFORM.tar.gz"
export PATH="$PWD/tensorfold-$V-$PLATFORM/bin:$PATH"
tensorfold() { command tensorfold-native "$@"; }
tensorfold pull mlx-community/Qwen3.5-2B-MLX-4bit@93760be4f1f69842a46bc13dbdc0f19e291392a3
tensorfold serve mlx-community/Qwen3.5-2B-MLX-4bit --name local-model --context 8192 --no-thinking
```

### Send a request

Leave the server running. In another terminal, on either platform:

```sh
curl -fsS http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"local-model","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":128,"temperature":0}'
```

The API also serves completions, Responses and Anthropic Messages. `/v1/models` lists the served name.

## Docs

- [Run guide](docs/run-guide.md): platform setup, requests, parallel requests, exact decoding, context compaction,
  Cortheon mode, every serve flag, the Mac app and TensorFold Starlight.
- [Supported models](docs/supported-models.md): which checkpoints run on which machines, models on disk and model discovery.
- [Living Weights (experimental)](docs/living-weights.md): teach a running model new facts in its own weights.
- [Installation runbook](RUNBOOK.md): macOS and CUDA installation, paired Metal serving and troubleshooting.
- [Windows](docs/WINDOWS.md), [TensorFold Web](docs/web.md) and [the Mac app](app/README.md).
- [API](docs/api.md), [Cortheon mode](docs/cortheon.md) and [known issues](docs/known-issues.md).
- [Evaluating TensorFold](docs/evaluating.md) and [community benchmarks](docs/community-benchmarks.md).
- [Usable memory profiles](docs/memory-profiling.md): how to record applied budgets, context, cache state and process peaks.
- [Contributing](CONTRIBUTING.md): ATC and agents, builds, tests and the exactness and performance gates.
- [Changes](CHANGELOG.md), [1.0.0 release notes](docs/release-notes-1.0.0.md) and [the Zig engine preview](docs/zig-preview.md).
- TensorFold is Apache-2.0: [LICENSE](LICENSE), [NOTICE](NOTICE), [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [SECURITY.md](SECURITY.md).

## Contributors

Thank you to everyone who has sent TensorFold a pull request, a measurement or a bug report:

[@0mao0](https://github.com/0mao0),
[@321sssrt-bit](https://github.com/321sssrt-bit),
[@AbrahamCete](https://github.com/AbrahamCete),
[@aditya1503](https://github.com/aditya1503),
[@AdrianBinDC](https://github.com/AdrianBinDC),
[@ajensenwaud](https://github.com/ajensenwaud),
[@akol1](https://github.com/akol1),
[@alenperic](https://github.com/alenperic),
[@Alexbob0](https://github.com/Alexbob0),
[@alexderz](https://github.com/alexderz),
[@anvilsong](https://github.com/anvilsong),
[@Arminova](https://github.com/Arminova),
[@b-ostrov](https://github.com/b-ostrov),
[@BadAd84](https://github.com/BadAd84),
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
[@CanadaApollo6](https://github.com/CanadaApollo6),
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
[@diwakergupta](https://github.com/diwakergupta),
[@drowzeys](https://github.com/drowzeys),
[@easthambuilds](https://github.com/easthambuilds),
[@ecohash-co](https://github.com/ecohash-co),
[@edurdias](https://github.com/edurdias),
[@eleqtrizit](https://github.com/eleqtrizit),
[@ericlsimplifi](https://github.com/ericlsimplifi),
[@EugeneClaw](https://github.com/EugeneClaw),
[@fche](https://github.com/fche),
[@feni6](https://github.com/feni6),
[@florianbrede-ayet](https://github.com/florianbrede-ayet),
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
[@GustavBlack](https://github.com/GustavBlack),
[@harrisonfriia](https://github.com/harrisonfriia),
[@haxudev](https://github.com/haxudev),
[@heitke](https://github.com/heitke),
[@hichaiuse](https://github.com/hichaiuse),
[@ivanfioravanti](https://github.com/ivanfioravanti),
[@jasontitus](https://github.com/jasontitus),
[@javierbits](https://github.com/javierbits),
[@jayleaton](https://github.com/jayleaton),
[@jeffpeng3](https://github.com/jeffpeng3),
[@jeidbugs404](https://github.com/jeidbugs404),
[@Jenpo](https://github.com/Jenpo),
[@jerryfane](https://github.com/jerryfane),
[@jetnet](https://github.com/jetnet),
[@jishnuvenugopal](https://github.com/jishnuvenugopal),
[@jkuepker](https://github.com/jkuepker),
[@joeynyc](https://github.com/joeynyc),
[@johnymoo](https://github.com/johnymoo),
[@JordiPosthumus](https://github.com/JordiPosthumus),
[@joshuaswarren](https://github.com/joshuaswarren),
[@JRaxworthy](https://github.com/JRaxworthy),
[@jrb-ai](https://github.com/jrb-ai),
[@jschmied](https://github.com/jschmied),
[@juliankang4](https://github.com/juliankang4),
[@justincasey](https://github.com/justincasey),
[@kelchm](https://github.com/kelchm),
[@kieranscotney](https://github.com/kieranscotney),
[@kingjamez](https://github.com/kingjamez),
[@kky42](https://github.com/kky42),
[@laudney](https://github.com/laudney),
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
[@mrv777](https://github.com/mrv777),
[@MV10](https://github.com/MV10),
[@NeoAiLabs](https://github.com/NeoAiLabs),
[@Nipale-ai](https://github.com/Nipale-ai),
[@nmcbride](https://github.com/nmcbride),
[@nood-co1](https://github.com/nood-co1),
[@nullburn](https://github.com/nullburn),
[@olexale](https://github.com/olexale),
[@omar16100](https://github.com/omar16100),
[@optimisme](https://github.com/optimisme),
[@otarkhan](https://github.com/otarkhan),
[@outcastofmusic](https://github.com/outcastofmusic),
[@paragontasx](https://github.com/paragontasx),
[@peacockesq](https://github.com/peacockesq),
[@philip-pentatonic](https://github.com/philip-pentatonic),
[@plotarmordev](https://github.com/plotarmordev),
[@pmeenan](https://github.com/pmeenan),
[@ppardus](https://github.com/ppardus),
[@pulseandthread](https://github.com/pulseandthread),
[@quigles1977](https://github.com/quigles1977),
[@rafafortes](https://github.com/rafafortes),
[@raymondkpwong](https://github.com/raymondkpwong),
[@robertpitt](https://github.com/robertpitt),
[@rootsbymenda](https://github.com/rootsbymenda),
[@RoscoeTT](https://github.com/RoscoeTT),
[@RS91Dev](https://github.com/RS91Dev),
[@russellmm](https://github.com/russellmm),
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
[@Stephane-Eden](https://github.com/Stephane-Eden),
[@styles01](https://github.com/styles01),
[@SvangenStudios](https://github.com/SvangenStudios),
[@swbiggart](https://github.com/swbiggart),
[@SxMShaDoW](https://github.com/SxMShaDoW),
[@sxuff](https://github.com/sxuff),
[@taussoe](https://github.com/taussoe),
[@tfolkman](https://github.com/tfolkman),
[@ThinkOffApp](https://github.com/ThinkOffApp),
[@ThomasHalwax](https://github.com/ThomasHalwax),
[@Thotheris](https://github.com/Thotheris),
[@tinyapps](https://github.com/tinyapps),
[@tolewis](https://github.com/tolewis),
[@tomByrer](https://github.com/tomByrer),
[@tonydehnke](https://github.com/tonydehnke),
[@tournierjc](https://github.com/tournierjc),
[@tovereast](https://github.com/tovereast),
[@tpischke](https://github.com/tpischke),
[@urtho](https://github.com/urtho),
[@valacroix](https://github.com/valacroix),
[@vcruz305](https://github.com/vcruz305),
[@vinicius-symetrix](https://github.com/vinicius-symetrix),
[@wojo](https://github.com/wojo),
[@xjqx2z](https://github.com/xjqx2z),
[@yguanrong](https://github.com/yguanrong),
[@Yuepixel](https://github.com/Yuepixel),
[@YvesLaRose](https://github.com/YvesLaRose),
[@ZackSample](https://github.com/ZackSample),
[@zweck](https://github.com/zweck).

[docs/credits.md](docs/credits.md) names whose Python engine work each part of the native engine carries.
