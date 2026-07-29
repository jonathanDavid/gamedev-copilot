import puppeteer from 'puppeteer-core';
import { readFileSync } from 'node:fs';
const OUT = 'C:/Users/Home/AppData/Local/Temp/claude/c--Users-Home-Documents-Personal--Portafolio/94769d44-240e-4d5a-bb0d-c3919b73e840/scratchpad/';
const raw = readFileSync('/tmp/copilot_session.txt', 'utf8');
// puppeteer-core installed in routegraph-web; reuse via relative import? No — chrome direct here needs puppeteer-core locally.
