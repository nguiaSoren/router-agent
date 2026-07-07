// Render github_card.html → github_card.png (1280x640 @2x = 2560x1280, GitHub social preview).
const { chromium } = require('playwright');
const path = require('path');
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1280, height: 640 }, deviceScaleFactor: 2 });
  await page.goto('file://' + path.join(__dirname, 'github_card.html'), { waitUntil: 'networkidle' });
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(600);
  await (await page.$('.card')).screenshot({ path: path.join(__dirname, 'out', 'github_card.png') });
  await browser.close();
  console.log('done → out/github_card.png');
})().catch(e => { console.error(e); process.exit(1); });
