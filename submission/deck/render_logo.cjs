const { chromium } = require('playwright');
const path = require('path');
(async () => {
  const b = await chromium.launch();
  const p = await b.newPage({ viewport: { width: 512, height: 512 }, deviceScaleFactor: 2 });
  await p.goto('file://' + path.join(__dirname, 'github_logo.html'), { waitUntil: 'networkidle' });
  await p.waitForTimeout(300);
  await (await p.$('.logo')).screenshot({ path: path.join(__dirname, 'out', 'logo.png') });
  await b.close(); console.log('done → out/logo.png');
})().catch(e => { console.error(e); process.exit(1); });
