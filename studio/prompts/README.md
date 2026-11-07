# Prompts

- `baker-image.txt` and `baker-video.txt`: the pair behind the samples. The first describes the picture, the second
  what happens in the clip, in MiniMax's image-to-video prompt format with a spoken line.
- `black-mirror-scene.txt`: "BLACK MIRROR: FINAL REFLECTION" by @itxabdullaa on X (onlyprompts.ai X collection), a
  text-to-video test prompt for `scripts/video.sh`.
- `baker-long-speech.txt`: the same baker speaking a 37-word passage, for 15 second clips (`FRAMES=362`); a test of
  sustained speech.
- `singer-image.txt`, `singer-video.txt` (15 s, `FRAMES=362`) and `singer-video-8s.txt` (`FRAMES=192`): an
  unaccompanied singer, the test that showed the adapter's sound at its worst.
