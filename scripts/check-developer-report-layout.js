// Render the Developer Experience Rmd first. Uses the same Playwright/Edge
// installation as capture-report-screenshots.js; no browser download is needed.
// Usage: node scripts/check-developer-report-layout.js [screenshot-directory]
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { chromium } = require('playwright');

const repo = path.resolve(__dirname, '..');
const html = path.join(repo, 'examples', 'utility-r',
  'github-copilot-developer-productivity-simulation.html');
const output = process.argv[2] ? path.resolve(process.argv[2]) : null;
const pages = ['overview', 'collaboration', 'focus', 'after-hours',
  'ai-use-and-query-coverage', 'github-copilot-breakdowns',
  'working-patterns-and-evaluation', 'appendix'];
const sizes = [[1440, 900], [1024, 768], [768, 1024], [390, 844]];

async function selectPage(page, id, mobile) {
  if (mobile && !(await page.locator('#navbar').isVisible())) {
    await page.locator('.navbar-toggle').click();
    await page.waitForFunction(() => document.getElementById('navbar').classList.contains('in'));
  }
  const link = page.locator(`.navbar-nav a[href="#${id}"]`);
  if (!(await link.isVisible())) {
    await page.locator('.navbar-nav > li.dropdown > a').click();
  }
  await link.click();
  await page.locator(`#${id}`).waitFor({ state: 'visible' });
  if (mobile) await page.locator('#navbar').waitFor({ state: 'hidden' });
  await page.waitForTimeout(400);
}

async function geometry(page, id) {
  return page.evaluate((pageId) => {
    const scope = document.getElementById(pageId);
    const visible = (element) => element.getBoundingClientRect().width > 0 &&
      !element.closest('details:not([open])');
    const charts = Array.from(scope.querySelectorAll('.report-chart img')).filter(visible);
    const overflow = Array.from(scope.querySelectorAll(
      '.chart-wrapper, .chart-stage, .report-chart, .data-table, .kpi, .pillar-grid a'
    )).filter(visible).filter((element) => {
      const rect = element.getBoundingClientRect();
      return rect.left < -1 || rect.right > window.innerWidth + 1 ||
        element.scrollWidth > element.clientWidth + 2;
    }).map((element) => element.className);
    const cards = Array.from(scope.querySelectorAll('.chart-wrapper')).filter(visible)
      .map((element) => element.getBoundingClientRect());
    return {
      overflow,
      pageOverflow: document.documentElement.scrollWidth > window.innerWidth + 1,
      cardsOverlap: cards.some((rect, i) => i > 0 && rect.top < cards[i - 1].bottom - 1),
      brokenCharts: charts.filter((image) => !image.complete || image.naturalWidth === 0).length,
      firstChartTop: charts.length ? charts[0].getBoundingClientRect().top : null,
      embeddedCharts: charts.every((image) => image.currentSrc.startsWith('data:image/svg+xml')),
      navbarHeight: document.querySelector('.navbar').getBoundingClientRect().height,
      navRows: new Set(Array.from(document.querySelectorAll('.navbar-nav > li'))
        .map((element) => Math.round(element.getBoundingClientRect().top))).size,
      narrowCaptions: window.innerWidth < 768 && Array.from(scope.querySelectorAll('.data-table caption'))
        .filter(visible).some((caption) => caption.getBoundingClientRect().width <
          caption.closest('table').getBoundingClientRect().width * .9),
      visiblePanelWidths: charts.map((image) => Math.round(image.getBoundingClientRect().width)),
    };
  }, id);
}

(async () => {
  if (output) fs.mkdirSync(output, { recursive: true });
  const browser = await chromium.launch({ channel: 'msedge' });
  const failures = [];
  const measurements = {};
  const check = (condition, label) => { if (!condition) failures.push(label); };
  try {
    for (const [width, height] of sizes) {
      const context = await browser.newContext({ viewport: { width, height } });
      await context.setOffline(true);
      const page = await context.newPage();
      page.on('pageerror', (error) => failures.push(`${width}px script error: ${error.message}`));
      await page.goto(pathToFileURL(html).href, { waitUntil: 'load' });
      await page.waitForSelector('.navbar-nav a[data-toggle="tab"]', { state: 'attached' });
      assert.equal(await page.locator('.navbar-nav > li').count(), 6);
      assert.equal(await page.locator('.navbar-nav a[data-toggle="tab"]').count(), 8);
      for (const id of pages) {
        await selectPage(page, id, width < 768);
        const initial = await geometry(page, id);
        measurements[`${width}-${id}`] = initial;
        check(!initial.pageOverflow && !initial.overflow.length, `${width}px ${id}: overflow ${initial.overflow}`);
        check(!initial.cardsOverlap, `${width}px ${id}: overlapping cards`);
        check(!initial.brokenCharts && initial.embeddedCharts, `${width}px ${id}: broken/external charts`);
        if (width >= 768) {
          check(initial.navRows === 1 && initial.navbarHeight < 70, `${width}px ${id}: navbar wraps`);
        }
        if (['collaboration', 'focus', 'after-hours', 'ai-use-and-query-coverage'].includes(id)) {
          check(initial.firstChartTop !== null && initial.firstChartTop < height - 80,
            `${width}px ${id}: no meaningful chart area in first viewport`);
        }
        if (output && width !== 768) {
          await page.screenshot({ path: path.join(output, `${width}-${id}.png`) });
        }
        if (id === 'overview') {
          check(await page.locator('#overview .kpi').count() === 4, 'Overview must have four KPIs');
          check(await page.locator('#overview .pillar-grid a').count() === 4, 'Pillar cards must be links');
          const rows = await page.locator('#overview .kpi').evaluateAll((elements) => {
            const counts = new Map();
            elements.forEach((element) => {
              const top = Math.round(element.getBoundingClientRect().top);
              counts.set(top, (counts.get(top) || 0) + 1);
            });
            return [...counts.values()];
          });
          check(rows.every((n) => n === rows[0]), `${width}px overview: stranded KPI`);
          const pillarTops = await page.locator('#overview .pillar-grid a').evaluateAll((elements) =>
            elements.map((element) => Math.round(element.getBoundingClientRect().top)));
          if (width >= 1024) check(new Set(pillarTops).size === 1, `${width}px: pillar links must occupy one row`);
        }
        await page.locator(`#${id} details`).evaluateAll((details) => {
          details.forEach((element) => { element.open = true; });
        });
        const expanded = await geometry(page, id);
        check(!expanded.pageOverflow && !expanded.overflow.length && !expanded.cardsOverlap,
          `${width}px ${id}: expanded-content overflow/overlap ${expanded.overflow}`);
        check(!expanded.brokenCharts, `${width}px ${id}: broken expanded chart`);
        check(!expanded.narrowCaptions, `${width}px ${id}: cramped table caption`);
        await page.locator(`#${id} details`).evaluateAll((details) => {
          details.forEach((element) => { element.open = false; });
        });
      }
      await selectPage(page, 'overview', width < 768);
      const summary = page.locator('#overview details > summary').first();
      await summary.focus();
      await page.keyboard.press('Enter');
      check(await summary.evaluate((element) => element.parentElement.open), `${width}px: keyboard disclosure`);
      check(await summary.evaluate((element) => parseFloat(getComputedStyle(element).outlineWidth) >= 2),
        `${width}px: visible keyboard focus`);
      if (output) {
        await page.locator('#overview .data-table').first().evaluate((element) => {
          element.scrollIntoView({ block: 'start' });
          window.scrollBy(0, -80);
        });
        await page.screenshot({ path: path.join(output, `${width}-expanded-table.png`) });
      }
      await page.keyboard.press('Enter');
      const shortcut = page.locator('#overview .pillar-grid a[href="#focus"]');
      await shortcut.focus();
      await page.keyboard.press('Enter');
      await page.locator('#focus').waitFor({ state: 'visible' });
      check(await page.locator('#focus .chart-title').evaluate((heading) => heading === document.activeElement),
        `${width}px: shortcut must transfer focus to destination`);
      if (width < 768) {
        await page.locator('.navbar-toggle').click();
        await page.waitForFunction(() => document.getElementById('navbar').classList.contains('in'));
      }
      const more = page.locator('.navbar-nav > li.dropdown > a');
      await more.focus();
      await page.keyboard.press('Enter');
      check(await page.locator('.navbar-nav a[href="#appendix"]').isVisible(), `${width}px: More keyboard expansion`);
      await page.keyboard.press('ArrowDown');
      check(await page.evaluate(() => document.activeElement.closest('.dropdown-menu') !== null),
        `${width}px: More arrow-key navigation`);
      await page.keyboard.press('Escape');
      if (width < 768) {
        await page.locator('.navbar-nav a[href="#overview"]').click();
        await page.locator('#navbar').waitFor({ state: 'hidden' });
        await page.locator('.navbar-toggle').click();
        check(await page.locator('#navbar').evaluate((element) => element.classList.contains('collapsing')),
          'Fast menu selection must exercise an opening transition');
        await page.locator('.navbar-nav a[href="#collaboration"]').evaluate((link) => link.click());
        await page.locator('#navbar').waitFor({ state: 'hidden' });
        check(await page.locator('#collaboration').isVisible(), 'Fast menu selection must open its destination');
      }
      await context.close();
    }
  } finally {
    await browser.close();
  }
  if (output) fs.writeFileSync(path.join(output, 'measurements.json'), JSON.stringify(measurements, null, 2));
  assert.deepEqual(failures, [], failures.join('\n'));
  console.log(`Layout and keyboard checks passed: ${pages.length} pages at ${sizes.length} viewports.`);
})().catch((error) => { console.error(error); process.exitCode = 1; });
