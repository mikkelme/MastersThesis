#!/bin/bash
# Bash script for copying the content of 'outbox'
# to cluster node 'bigfacet'

scp -r ./outbox/* bigfacet:./inbox

# echo Hello World!