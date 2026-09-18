// Regenerate the docs-site report screenshots from the rendered flexdashboards.
//
// Prerequisites (Playwright is not a committed dependency):
//   npm install --no-save playwright
//
// Drives the system Microsoft Edge install via Playwright's `msedge` channel,
// so no separate browser download is required.
//
// Usage, from the repository root:
//   node scripts/capture-report-screenshots.js
//   node scripts/capture-report-screenshots.js --github-only
//
// Render both reports first, otherwise the screenshots capture stale charts:
//   cd examples/utility-r
//   Rscript -e "rmarkdown::render('copilot-consumption-ways-of-working-simulation.Rmd')"
//   Rscript render-github-developer-experience.R
const { chromium } = require('playwright');
const path = require('path');

const REPO = path.resolve(__dirname, '..');
const UTILITY = path.join(REPO, 'examples', 'utility-r');
const OUT = path.join(REPO, 'assets', 'images', 'reports');

const SHOTS = [
  {
    html: 'copilot-consumption-ways-of-working-simulation.html',
    page: 'overview',
    file: 'copilot-consumption-overview.png',
  },
  {
    html: 'copilot-consumption-ways-of-working-simulation.html',
    page: 'product-usage-mix',
    file: 'copilot-consumption-credit-distribution.png',
  },
  {
    html: 'github-copilot-developer-productivity-simulation.html',
    page: 'ai-use-and-query-coverage',
    file: 'github-copilot-devex-ai-use.png',
  },
  {
    html: 'github-copilot-developer-productivity-simulation.html',
    page: 'focus',
    file: 'github-copilot-devex-focus.png',
  },
];

(async () => {
  const browser = await chromium.launch({ channel: 'msedge' });
  try {
    const context = await browser.newContext({
      viewport: { width: 1440, height: 900 },
      deviceScaleFactor: 1,
    });
    const page = await context.newPage();

    const shots = process.argv.includes('--github-only')
      ? SHOTS.filter((shot) => shot.html.startsWith('github-')) : SHOTS;
    for (const shot of shots) {
      const url = 'file:///' + path.join(UTILITY, shot.html).replace(/\\/g, '/');
      await page.goto(url, { waitUntil: 'load' });
      await page.waitForSelector('.navbar-nav a', { timeout: 60000 });

      const clicked = await page.evaluate((id) => {
        const links = Array.from(document.querySelectorAll('.navbar-nav a'));
        const target = links.find((a) => (a.getAttribute('href') || '').replace('#', '') === id);
        if (!target) return false;
        target.click();
        return true;
      }, shot.page);

      if (!clicked) {
        const available = await page.evaluate(() =>
          Array.from(document.querySelectorAll('.navbar-nav a')).map((a) => a.getAttribute('href'))
        );
        throw new Error(`Page "${shot.page}" not found in ${shot.html}. Available: ${available.join(', ')}`);
      }

      // Let the tab switch settle and any chart images paint.
      await page.waitForTimeout(2500);
      // flexdashboard can retain scroll position across tab switches, which
      // clips the first card's heading. Reset before capturing.
      await page.evaluate(() => {
        window.scrollTo(0, 0);
        document.querySelectorAll('.section.level1').forEach((s) => { s.scrollTop = 0; });
        document.querySelectorAll('.dashboard-column, .chart-wrapper, .chart-stage').forEach((s) => { s.scrollTop = 0; });
      });
      await page.waitForTimeout(600);
      await page.screenshot({ path: path.join(OUT, shot.file) });
      console.log(`captured ${shot.file}`);
    }
  } finally {
    await browser.close();
  }
})().catch((error) => { console.error(error); process.exitCode = 1; });
