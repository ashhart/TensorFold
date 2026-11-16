#!/bin/bash
# usage (on host-a): abba.sh SUITE CTX LABEL:TREE... -- [serve args]; one launch a label in order, suite-SUITE.sh in each.
suite=$1 ctx=$2; shift 2
labels=(); while [ $# -gt 0 ] && [ "$1" != -- ]; do labels+=("$1"); shift; done; shift
for lt in "${labels[@]}"; do
  label=${lt%%:*}; tree=${lt##*:}
  ~/glm-perf/up.sh $tree $label $ctx -- "$@" || { ~/glm-perf/down.sh $label; continue; }
  (cd ~/glm-perf/out/$label && bash ~/glm-perf/suite-$suite.sh $tree > suite.log 2>&1)
  echo "$(date '+%F %T') $label suite $(grep -h '^RESULT' ~/glm-perf/out/$label/suite.log | tr '\n' ' ')" >> ~/glm-perf/runs.log
  ~/glm-perf/down.sh $label
done
echo "$(date '+%F %T') abba END" >> ~/glm-perf/runs.log
