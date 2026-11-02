# Third-party notices

Starlight, Starlight Node and TensorFold Connect are part of TensorFold, which is licensed under Apache-2.0. Their binaries
include the Go standard library and the modules below. Each licence type and copyright line is taken from the
module's own licence file, NOTICE file or source headers at the version in `go.sum`; the full licence texts are in
each module's source. Container images built from `deploy/starlight/Dockerfile` also contain the distroless base
image's files, which carry their own licences.

| Module | Version | Licence | Used by |
| --- | --- | --- | --- |
| Go standard library | 1.27 | BSD-3-Clause | starlight, starlight-node, tensorfold-connect |
| `fyne.io/systray` | v1.12.2 | Apache-2.0 | tensorfold-connect |
| `github.com/beevik/etree` | v1.5.0 | BSD-2-Clause | starlight |
| `github.com/beorn7/perks` | v1.0.1 | MIT | starlight |
| `github.com/cespare/xxhash/v2` | v2.3.0 | MIT | starlight |
| `github.com/coreos/go-oidc/v3` | v3.21.0 | Apache-2.0 | starlight |
| `github.com/crewjam/saml` | v0.5.1 | BSD-2-Clause | starlight |
| `github.com/go-jose/go-jose/v4` | v4.1.7 | Apache-2.0 | starlight |
| `github.com/godbus/dbus/v5` | v5.1.0 | BSD-2-Clause | tensorfold-connect |
| `github.com/hashicorp/mdns` | v1.0.7 | MIT | starlight, tensorfold-connect |
| `github.com/jackc/pgpassfile` | v1.0.0 | MIT | starlight |
| `github.com/jackc/pgservicefile` | v0.0.0-20240606120523-5a60cdf6a761 | MIT | starlight |
| `github.com/jackc/pgx/v5` | v5.12.0 | MIT | starlight |
| `github.com/jackc/puddle/v2` | v2.2.3 | MIT | starlight |
| `github.com/jonboulle/clockwork` | v0.2.2 | Apache-2.0 | starlight |
| `github.com/mattermost/xml-roundtrip-validator` | v0.1.0 | Apache-2.0 | starlight |
| `github.com/miekg/dns` | v1.1.72 | BSD-3-Clause | starlight, tensorfold-connect |
| `github.com/munnerz/goautoneg` | v0.0.0-20191010083416-a7dc8b61c822 | BSD-3-Clause | starlight |
| `github.com/prometheus/client_golang` | v1.25.0 | Apache-2.0 | starlight |
| `github.com/prometheus/client_model` | v0.6.3 | Apache-2.0 | starlight |
| `github.com/prometheus/common` | v0.72.0 | Apache-2.0 | starlight |
| `github.com/prometheus/procfs` | v0.22.0 | Apache-2.0 | starlight (Linux only) |
| `github.com/russellhaering/goxmldsig` | v1.4.0 | Apache-2.0 | starlight |
| `go.yaml.in/yaml/v3` | v3.0.6 | Apache-2.0 and MIT | starlight |
| `golang.org/x/crypto` | v0.58.0 | BSD-3-Clause | starlight |
| `golang.org/x/net` | v0.62.0 | BSD-3-Clause | starlight, tensorfold-connect |
| `golang.org/x/oauth2` | v0.37.0 | BSD-3-Clause | starlight |
| `golang.org/x/sync` | v0.24.0 | BSD-3-Clause | starlight |
| `golang.org/x/sys` | v0.49.0 | BSD-3-Clause | starlight, starlight-node (Windows only), tensorfold-connect |
| `golang.org/x/text` | v0.43.0 | BSD-3-Clause | starlight |
| `google.golang.org/protobuf` | v1.36.12 | BSD-3-Clause | starlight |

## Copyright lines and NOTICE files

Apache-2.0 section 4(d) requires the NOTICE files of Apache-2.0 modules to travel with the binaries; they are
reproduced in full.

### Go standard library

BSD-3-Clause (`LICENSE` in the Go distribution).

```text
Copyright 2009 The Go Authors.
```

### fyne.io/systray v1.12.2

Apache-2.0 (`LICENSE`).

```text
Copyright 2014 Brave New Software Project, Inc.
```

### github.com/beevik/etree v1.5.0

BSD-2-Clause (`LICENSE`).

```text
Copyright 2015-2024 Brett Vickers. All rights reserved.
```

### github.com/beorn7/perks v1.0.1

MIT (`LICENSE`).

```text
Copyright (C) 2013 Blake Mizerany
```

### github.com/cespare/xxhash/v2 v2.3.0

MIT (`LICENSE.txt`).

```text
Copyright (c) 2016 Caleb Spare
```

### github.com/coreos/go-oidc/v3 v3.21.0

Apache-2.0 (`LICENSE`).

NOTICE:

```text
CoreOS Project
Copyright 2014 CoreOS, Inc

This product includes software developed at CoreOS, Inc.
(http://www.coreos.com/).
```

### github.com/crewjam/saml v0.5.1

BSD-2-Clause (`LICENSE`).

```text
Copyright (c) 2015, Ross Kinder
```

### github.com/go-jose/go-jose/v4 v4.1.7

Apache-2.0 (`LICENSE`).

The licence file has no copyright line; these are from the module's source headers.

```text
Copyright 2014 Square Inc.
Copyright 2016 Zbigniew Mandziejewicz
```

### github.com/godbus/dbus/v5 v5.1.0

BSD-2-Clause (`LICENSE`).

```text
Copyright (c) 2013, Georg Reinke (<guelfey at gmail dot com>), Google
```

### github.com/hashicorp/mdns v1.0.7

MIT (`LICENSE`).

```text
Copyright IBM Corp. 2014, 2026
```

### github.com/jackc/pgpassfile v1.0.0

MIT (`LICENSE`).

```text
Copyright (c) 2019 Jack Christensen
```

### github.com/jackc/pgservicefile v0.0.0-20240606120523-5a60cdf6a761

MIT (`LICENSE`).

```text
Copyright (c) 2020 Jack Christensen
```

### github.com/jackc/pgx/v5 v5.12.0

MIT (`LICENSE`).

```text
Copyright (c) 2013-2021 Jack Christensen
```

### github.com/jackc/puddle/v2 v2.2.3

MIT (`LICENSE`).

```text
Copyright (c) 2018 Jack Christensen
```

### github.com/jonboulle/clockwork v0.2.2

Apache-2.0 (`LICENSE`).

No copyright line appears in the licence file, a NOTICE file or the source headers.

### github.com/mattermost/xml-roundtrip-validator v0.1.0

Apache-2.0 (`LICENSE.txt`).

No copyright line appears in the licence file, a NOTICE file or the source headers.

### github.com/miekg/dns v1.1.72

BSD-3-Clause (`LICENSE`).

```text
Copyright (c) 2009, The Go Authors. Extensions copyright (c) 2011, Miek Gieben.
```

### github.com/munnerz/goautoneg v0.0.0-20191010083416-a7dc8b61c822

BSD-3-Clause (`LICENSE`).

```text
Copyright (c) 2011, Open Knowledge Foundation Ltd.
```

### github.com/prometheus/client_golang v1.25.0

Apache-2.0 (`LICENSE`).

NOTICE:

```text
Prometheus instrumentation library for Go applications
Copyright 2012-2015 The Prometheus Authors

This product includes software developed at
SoundCloud Ltd. (http://soundcloud.com/).


The following components are included in this product:

perks - a fork of https://github.com/bmizerany/perks
https://github.com/beorn7/perks
Copyright 2013-2015 Blake Mizerany, Björn Rabenstein
See https://github.com/beorn7/perks/blob/master/README.md for license details.

Go support for Protocol Buffers - Google's data interchange format
http://github.com/golang/protobuf/
Copyright 2010 The Go Authors
See source code for license details.
```

### github.com/prometheus/client_model v0.6.3

Apache-2.0 (`LICENSE`).

NOTICE:

```text
Data model artifacts for Prometheus.
Copyright 2012-2015 The Prometheus Authors

This product includes software developed at
SoundCloud Ltd. (http://soundcloud.com/).
```

### github.com/prometheus/common v0.72.0

Apache-2.0 (`LICENSE`).

NOTICE:

```text
Common libraries shared by Prometheus Go components.
Copyright 2015 The Prometheus Authors

This product includes software developed at
SoundCloud Ltd. (http://soundcloud.com/).
```

### github.com/prometheus/procfs v0.22.0

Apache-2.0 (`LICENSE`).

NOTICE:

```text
procfs provides functions to retrieve system, kernel and process
metrics from the pseudo-filesystem proc.

Copyright 2014-2015 The Prometheus Authors

This product includes software developed at
SoundCloud Ltd. (http://soundcloud.com/).
```

### github.com/russellhaering/goxmldsig v1.4.0

Apache-2.0 (`LICENSE`).

No copyright line appears in the licence file, a NOTICE file or the source headers.

### go.yaml.in/yaml/v3 v3.0.6

MIT and Apache-2.0 (`LICENSE`). The files ported from libyaml (`apic.go`, `emitterc.go`, `parserc.go`,
`readerc.go`, `scannerc.go`, `writerc.go`, `yamlh.go`, `yamlprivateh.go`) are MIT; the remaining files are
Apache-2.0.

```text
Copyright (c) 2006-2010 Kirill Simonov
Copyright (c) 2006-2011 Kirill Simonov
Copyright (c) 2011-2019 Canonical Ltd
```

NOTICE:

```text
Copyright 2011-2016 Canonical Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

### golang.org/x/crypto v0.58.0

BSD-3-Clause (`LICENSE`).

```text
Copyright 2009 The Go Authors.
```

### golang.org/x/net v0.62.0

BSD-3-Clause (`LICENSE`).

```text
Copyright 2009 The Go Authors.
```

### golang.org/x/oauth2 v0.37.0

BSD-3-Clause (`LICENSE`).

```text
Copyright 2009 The Go Authors.
```

### golang.org/x/sync v0.24.0

BSD-3-Clause (`LICENSE`).

```text
Copyright 2009 The Go Authors.
```

### golang.org/x/sys v0.49.0

BSD-3-Clause (`LICENSE`).

```text
Copyright 2009 The Go Authors.
```

### golang.org/x/text v0.43.0

BSD-3-Clause (`LICENSE`).

```text
Copyright 2009 The Go Authors.
```

### google.golang.org/protobuf v1.36.12

BSD-3-Clause (`LICENSE`).

```text
Copyright (c) 2018 The Go Authors. All rights reserved.
```
