// Render deck.html → per-slide PNGs (2x) + a 16:9 PDF. Playwright/chromium.
const { chromium } = require('playwright');
const path = require('path');
const fs = require('fs');

const DIR = __dirname;
const OUT = path.join(DIR, 'out');
const W = 1920, H = 1080;

(async () => {
  fs.mkdirSync(OUT, { recursive: true });
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 2 });
  await page.goto('file://' + path.join(DIR, 'deck.html'), { waitUntil: 'networkidle' });
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(700);

  // per-slide PNGs
  const slides = await page.$$('.slide');
  for (let i = 0; i < slides.length; i++) {
    const id = await slides[i].getAttribute('id');
    const n = String(i + 1).padStart(2, '0');
    await slides[i].screenshot({ path: path.join(OUT, `${n}_${id}.png`) });
    console.log('slide', n, id);
  }
  // cover as a standalone 16:9 image for the submission cover field
  fs.copyFileSync(path.join(OUT, '01_s01.png'), path.join(OUT, 'cover.png'));

  // one-slide-per-page PDF, backgrounds preserved
  await page.addStyleTag({ content:
    '@page{size:1920px 1080px;margin:0} .slide{break-after:page} html,body{background:#000}' });
  await page.emulateMedia({ media: 'print' });
  await page.pdf({ path: path.join(OUT, 'TokenGolf_deck.pdf'),
    width: '1920px', height: '1080px', printBackground: true, pageRanges: '' });

  await browser.close();
  console.log('done →', OUT);
})().catch(e => { console.error(e); process.exit(1); });
