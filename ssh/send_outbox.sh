#!/bin/bash
# Bash script for copying the content of 'outbox'

scp -r ./outbox/* bigfacet:./inbox

# echo Hello World!
