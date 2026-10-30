// Build the Jekyll site first. Uses the same Playwright/Edge setup as
// check-developer-report-layout.js. Usage: node scripts/check-catalogue-dialogs.js [site-directory]
const assert = require('node:assert/strict');
const fs = require('node:fs/promises');
const http = require('node:http');
const path = require('node:path');
const { chromium } = require('playwright');

const site = path.resolve(process.argv[2] || path.join(__dirname, '..', '_site'));
const prefix = '/viva-insights-sample-code/';
const types = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
};

const server = http.createServer(async (request, response) => {
  try {
    const pathname = decodeURIComponent(new URL(request.url, 'http://localhost').pathname);
    if (!pathname.startsWith(prefix)) {
      response.writeHead(404).end();
      return;
    }
    let filename = path.resolve(site, pathname.slice(prefix.length));
    const relative = path.relative(site, filename);
    if (relative === '..' || relative.startsWith('..' + path.sep) || path.isAbsolute(relative)) {
      response.writeHead(403).end();
      return;
    }
    if ((await fs.stat(filename)).isDirectory()) filename = path.join(filename, 'index.html');
    const contents = await fs.readFile(filename);
    response.writeHead(200, { 'Content-Type': types[path.extname(filename)] || 'application/octet-stream' });
    response.end(contents);
  } catch (error) {
    if (error.code === 'ENOENT' || error.code === 'ENOTDIR') {
      response.writeHead(404).end();
    } else {
      console.error(error);
      response.writeHead(500).end();
    }
  }
});

(async () => {
  await new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', resolve);
  });
  const origin = `http://127.0.0.1:${server.address().port}`;
  const base = origin + prefix;
  let browser;
  try {
    browser = await chromium.launch({ channel: 'msedge' });
    const context = await browser.newContext({ viewport: { width: 320, height: 720 } });
    await context.route('**/*', (route) =>
      new URL(route.request().url()).origin === origin ? route.continue() : route.abort());
    const page = await context.newPage();
    const pageErrors = [];
    page.on('pageerror', (error) => pageErrors.push(error.message));

    await page.goto(base + 'frontier-analytics-prompts/', { waitUntil: 'load' });
    assert.equal(await page.locator('.prompt-card').count(), 10);
    assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
    assert.equal(await page.locator('.prompt-card-summary a').count(), 0);
    assert.match(await page.locator('[data-prompt-open="audit-parsing"]').locator('..').innerText(), /Agent Usage Analysis/);
    const prompt = page.locator('[data-prompt-open="segmentation"]');
    await prompt.click();
    const promptDialog = page.locator('.prompt-dialog[open]');
    const scrollTop = await promptDialog.evaluate((dialog) => {
      dialog.scrollTop = dialog.scrollHeight;
      return dialog.scrollTop;
    });
    assert.ok(scrollTop > 0, 'The mobile prompt dialog must be tall enough to scroll.');
    await promptDialog.locator('.prompt-dialog-close').evaluate((button) => button.click());
    await page.waitForFunction(() => !document.querySelector('.prompt-dialog[open]') && !location.hash);
    await prompt.click();
    const reopened = await promptDialog.evaluate((dialog) => {
      const title = dialog.querySelector('.prompt-details-title');
      const bounds = dialog.getBoundingClientRect();
      const rect = title.getBoundingClientRect();
      return {
        scrollTop: dialog.scrollTop,
        focused: document.activeElement === title,
        visible: rect.top >= bounds.top && rect.bottom <= bounds.bottom,
      };
    });
    assert.deepEqual(reopened, { scrollTop: 0, focused: true, visible: true });

    await page.goto(base + 'frontier-analytics-prompts/#prompt-segmentation', { waitUntil: 'load' });
    await page.waitForSelector('.prompt-dialog[open] [data-prompt-panel="segmentation"]');
    const noScript = await browser.newContext({ javaScriptEnabled: false, viewport: { width: 320, height: 720 } });
    await noScript.route('**/*', (route) =>
      new URL(route.request().url()).origin === origin ? route.continue() : route.abort());
    const fallbackPage = await noScript.newPage();
    await fallbackPage.goto(base + 'frontier-analytics-prompts/', { waitUntil: 'load' });
    assert.equal(await fallbackPage.locator('[data-prompt-details] [data-prompt-panel]').count(), 10);
    for (const id of ['copilot-adoption', 'query-dashboards', 'purview-augmentation']) {
      assert.equal(await fallbackPage.locator(`[id="${id}"]`).count(), 1);
    }
    await noScript.close();

    await page.goto(base + 'frontier-analytics-skills/#get-started', { waitUntil: 'load' });
    assert.equal(await page.locator('#get-started').count(), 1);
    assert.equal(await page.locator('.skill-dialog[open]').count(), 0);
    const opener = page.locator('[data-skill-open]').first();
    const slug = await opener.getAttribute('data-skill-open');
    await opener.click();
    const skillDialog = page.locator('.skill-dialog[open]');
    const overview = `#skill-${slug}-overview`;
    const results = `#skill-${slug}-results`;
    await skillDialog.locator(`[href="${overview}"]`).focus();
    await page.keyboard.press('Enter');
    await page.waitForFunction((hash) => location.hash === hash, overview);
    const headingState = await page.evaluate(() => {
      const heading = document.activeElement;
      return {
        section: heading.closest('.skill-section').id,
        focusVisible: heading.matches(':focus-visible'),
        ring: getComputedStyle(heading).boxShadow !== 'none',
      };
    });
    assert.deepEqual(headingState, { section: overview.slice(1), focusVisible: true, ring: true });
    await skillDialog.locator(`[href="${results}"]`).click();
    assert.equal(await page.evaluate(() => location.hash), results);
    await page.evaluate(() => history.back());
    await page.waitForFunction((hash) =>
      location.hash === hash && document.activeElement.closest('.skill-section')?.id === hash.slice(1), overview);
    await page.evaluate(() => history.forward());
    await page.waitForFunction((hash) =>
      location.hash === hash && document.activeElement.closest('.skill-section')?.id === hash.slice(1), results);
    await skillDialog.locator('.skill-dialog-close').click();
    await page.waitForFunction(() => !document.querySelector('.skill-dialog[open]') && location.hash === '#get-started');
    assert.equal(await page.evaluate(() => document.activeElement.getAttribute('data-skill-open')), slug);
    await page.evaluate(() => history.back());
    await page.waitForFunction((hash) => location.hash === hash && !!document.querySelector('.skill-dialog[open]'), overview);
    await page.evaluate(() => history.forward());
    await page.waitForFunction(() => location.hash === '#get-started' && !document.querySelector('.skill-dialog[open]'));
    await page.goto(base + 'frontier-analytics-skills/' + results, { waitUntil: 'load' });
    await page.waitForFunction((hash) =>
      !!document.querySelector('.skill-dialog[open]') &&
      document.activeElement.closest('.skill-section')?.id === hash.slice(1), results);
    await skillDialog.locator('.skill-dialog-close').click();
    await page.waitForFunction(() => !document.querySelector('.skill-dialog[open]') && !location.hash);
    assert.deepEqual(pageErrors, []);
    console.log('Catalogue mobile, fallback, focus and history checks passed.');
  } finally {
    if (browser) await browser.close();
    await new Promise((resolve, reject) => server.close((error) => error ? reject(error) : resolve()));
  }
})().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
