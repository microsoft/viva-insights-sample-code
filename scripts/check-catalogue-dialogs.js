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
    page.setDefaultTimeout(10000);
    const pageErrors = [];
    page.on('pageerror', (error) => pageErrors.push(error.message));

    for (const [pathname, title] of [
      ['frontier-analytics/', 'Frontier'],
      ['frontier-analytics-skills/', 'Agent Skills for Viva Insights'],
      ['frontier-analytics-prompts/', 'Viva Insights Prompt Library'],
      ['frontier-analytics-schemas/', 'Viva Insights Schema Documentation'],
    ]) {
      await page.goto(base + pathname, { waitUntil: 'load' });
      assert.equal(await page.locator('h1').count(), 1);
      assert.equal(await page.locator('h1').innerText(), title);
      assert.equal((await page.locator('.vi-page-body').innerText()).includes('\u2014'), false);
      if (pathname === 'frontier-analytics-skills/' || pathname === 'frontier-analytics-prompts/') {
        const catalogue = page.getByRole('link', { name: 'dashboard catalogue', exact: true });
        await catalogue.click();
        await page.waitForURL(base + 'copilot-dashboards/');
        assert.equal(await page.locator('[data-dashboard-open="business-leaders-ai-usage"]').count(), 1);
      }
    }

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

    await page.goto(base + 'frontier-analytics-prompts/#query-dashboards', { waitUntil: 'load' });
    await page.evaluate(() => history.replaceState({ existing: 'prompt' }, '', location.href));
    await prompt.click();
    await page.evaluate(() => history.back());
    await page.waitForFunction(() => location.hash === '#query-dashboards' && !document.querySelector('.prompt-dialog[open]'));
    await page.evaluate(() => history.forward());
    await page.waitForFunction(() => location.hash === '#prompt-segmentation' && !!document.querySelector('.prompt-dialog[open]'));
    await promptDialog.locator('.prompt-dialog-close').click();
    await page.waitForFunction(() => location.hash === '#query-dashboards' && !document.querySelector('.prompt-dialog[open]'));
    assert.deepEqual(await page.evaluate(() => history.state), { existing: 'prompt' });

    await page.goto(base + 'frontier-analytics-prompts/#prompt-segmentation', { waitUntil: 'load' });
    await page.waitForSelector('.prompt-dialog[open] [data-prompt-panel="segmentation"]');
    await promptDialog.locator('.prompt-dialog-close').click();
    await page.waitForFunction(() => !document.querySelector('.prompt-dialog[open]') && !location.hash);
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
    await page.evaluate(() => history.replaceState({ existing: 'skill' }, '', location.href));
    await opener.click();
    const skillDialog = page.locator('.skill-dialog[open]');
    await page.evaluate(() => history.back());
    await page.waitForFunction(() => location.hash === '#get-started' && !document.querySelector('.skill-dialog[open]'));
    await page.evaluate(() => history.forward());
    await page.waitForFunction((id) => location.hash === '#skill-' + id && !!document.querySelector('.skill-dialog[open]'), slug);
    await skillDialog.locator('.skill-dialog-close').click();
    await page.waitForFunction(() => !document.querySelector('.skill-dialog[open]') && location.hash === '#get-started');
    assert.deepEqual(await page.evaluate(() => history.state), { existing: 'skill' });
    await opener.click();
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

    for (const width of [320, 1280]) {
      await page.setViewportSize({ width, height: 720 });
      await page.goto(base, { waitUntil: 'load' });
      const navToggle = page.getByRole('button', { name: 'Toggle navigation' });
      if (await navToggle.isVisible()) await navToggle.click();
      await page.locator('[aria-controls="vi-mega-copilot"]').click();
      const navGuideLink = page.locator('#vi-mega-copilot a', { hasText: 'Business Leaders AI Usage' });
      await navGuideLink.focus();
      await page.keyboard.press('Enter');
      await page.waitForURL(base + 'copilot-dashboards/business-leaders-ai-usage/');
      assert.equal(await page.getByRole('heading', { name: 'Business Leaders AI Usage', exact: true }).isVisible(), true);
      assert.equal(await page.locator('#you-will-need').count(), 1);
      for (const pathname of ['copilot/', 'copilot-dashboards/', 'frontier-analytics/']) {
        await page.goto(base + pathname, { waitUntil: 'load' });
        assert.equal(await page.locator('.dashboard-card').count(), 3);
        assert.equal(await page.locator('[data-dashboard-details]').count(), 0);
        assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
        const navGuide = page.locator('#vi-mega-copilot a', { hasText: 'Business Leaders AI Usage' });
        assert.equal(await navGuide.getAttribute('href'), prefix + 'copilot-dashboards/business-leaders-ai-usage/');
        assert.equal(await page.locator('.dashboard-card img').count(), 3);
        assert.equal(await page.evaluate(() => {
          const ids = Array.from(document.querySelectorAll('[id]'), (element) => element.id);
          return ids.length === new Set(ids).size;
        }), true, 'Element IDs must be unique on ' + pathname);
        for (const card of await page.locator('.dashboard-card').all()) {
          const slug = await card.getAttribute('id');
          const guidePath = prefix + 'copilot-dashboards/' + slug + '/';
          assert.equal(await card.getByRole('link', { name: /^Full guide/ }).getAttribute('href'), guidePath);
          for (const menu of ['copilot', 'frontier']) {
            assert.equal(await page.locator(`#vi-mega-${menu} a[href="${guidePath}"]`).count(), 1);
          }
        }
        for (const trigger of await page.locator('[data-dashboard-open]').all()) {
          const slug = await trigger.getAttribute('data-dashboard-open');
          await trigger.focus();
          await page.keyboard.press('Enter');
          const dialog = page.locator('.dashboard-dialog[open]');
          assert.equal(await dialog.count(), 1);
          assert.equal(await dialog.getAttribute('aria-labelledby'), 'dashboard-title-' + slug);
          assert.equal(await dialog.locator('h3').evaluate((title) => title === document.activeElement), true);
          assert.equal(await dialog.getByRole('link', { name: 'View demo', exact: true }).count(), 1);
          assert.equal(await dialog.getByRole('link', { name: 'Get source', exact: true }).count(), 1);
          assert.equal(await dialog.getByRole('link', { name: 'Rendering and input requirements' }).count(), 1);
          assert.match(await dialog.innerText(), /Expected output:/);
          assert.match(await dialog.innerText(), /Using real exports:/);
          assert.match(await dialog.innerText(), /Work with a coding agent/);
          assert.match(await dialog.locator('pre').first().innerText(), /git clone/);
          assert.match(await dialog.locator('pre').nth(1).innerText(), /install\.packages\(c\('(?:\w+', ')*\w+'\)/);
          assert.match(await dialog.locator('pre').nth(2).innerText(), /Rscript.*(?:render-business-leaders-ai|reproduce)/);
          assert.ok(await dialog.locator('pre').count() >= 4);
          assert.equal(await dialog.evaluate((element) => element.scrollWidth <= element.clientWidth), true);
          for (const link of await dialog.locator('a').all()) {
            const url = new URL(await link.getAttribute('href'), base);
            if (url.origin !== origin) continue;
            const response = await context.request.get(url.href);
            assert.equal(response.status(), 200, 'Dashboard link must resolve: ' + url.href);
            if (url.hash) assert.ok((await response.text()).includes(`id="${url.hash.slice(1)}"`), url.href);
          }
          await dialog.locator('a').last().focus();
          await page.keyboard.press('Tab');
          assert.equal(await dialog.locator('.dashboard-dialog-close').evaluate((button) => button === document.activeElement), true);
          await page.keyboard.press('Escape');
          await page.waitForFunction(() => !document.querySelector('.dashboard-dialog[open]') && !location.hash);
          assert.equal(await trigger.evaluate((link) => link === document.activeElement), true);
        }
        const business = page.locator('[data-dashboard-open="business-leaders-ai-usage"]');
        await business.click();
        await page.evaluate(() => history.back());
        await page.waitForFunction(() => !document.querySelector('.dashboard-dialog[open]') && !location.hash);
        await page.evaluate(() => history.forward());
        await page.waitForSelector('.dashboard-dialog[open]');
        await page.locator('.dashboard-dialog[open] .dashboard-dialog-close').click();
        await page.waitForFunction(() => !document.querySelector('.dashboard-dialog[open]') && !location.hash);
        await business.click();
        await page.locator('.dashboard-dialog[open]').evaluate((dialog) => { dialog.scrollTop = dialog.scrollHeight; });
        await page.keyboard.press('Escape');
        await page.waitForFunction(() => !document.querySelector('.dashboard-dialog[open]'));
        await business.click();
        assert.equal(await page.locator('.dashboard-dialog[open]').evaluate((dialog) => dialog.scrollTop), 0);
        await page.mouse.click(2, 2);
        await page.waitForFunction(() => !document.querySelector('.dashboard-dialog[open]'));
        if (pathname === 'copilot-dashboards/') {
          for (const trigger of await page.locator('[data-dashboard-open]').all()) {
            const slug = await trigger.getAttribute('data-dashboard-open');
            await trigger.click();
            await page.locator('.dashboard-dialog[open]').getByRole('link', { name: 'Full dashboard details and caveats', exact: true }).click();
            await page.waitForURL(base + 'copilot-dashboards/' + slug + '/');
            assert.equal(await page.locator('h1').count(), 1);
            assert.equal(await page.locator('#you-will-need').count(), 1);
            assert.ok(await page.locator('.vi-page-body img').count() >= 2);
            assert.ok(await page.locator('.vi-page-body pre').count() >= 2);
            assert.equal(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
            for (const image of await page.locator('.vi-page-body img').all()) {
              await image.scrollIntoViewIfNeeded();
              await image.evaluate((img) => img.decode());
              assert.equal(await image.evaluate((img) => img.complete && img.naturalWidth > 0), true);
            }
            for (const link of await page.locator('.vi-page-body a').all()) {
              const url = new URL(await link.getAttribute('href'), base);
              if (url.origin !== origin) continue;
              const response = await context.request.get(url.href);
              assert.equal(response.status(), 200, 'Guide link must resolve: ' + url.href);
              if (url.hash) assert.ok((await response.text()).includes(`id="${url.hash.slice(1)}"`), url.href);
            }
            await page.goto(base + 'copilot-dashboards/', { waitUntil: 'load' });
          }
        }
      }
    }

    await page.goto(base + 'copilot-dashboards/#dashboard-business-leaders-ai-usage', { waitUntil: 'load' });
    await page.waitForSelector('.dashboard-dialog[open]');
    await page.locator('.dashboard-dialog[open]').getByRole('link', { name: 'Full dashboard details and caveats' }).click();
    await page.waitForURL(base + 'copilot-dashboards/business-leaders-ai-usage/');
    assert.equal(await page.locator('#you-will-need').isVisible(), true);

    for (const disabled of ['javascript', 'dialog']) {
      const fallback = await browser.newContext({ javaScriptEnabled: disabled !== 'javascript', viewport: { width: 320, height: 720 } });
      await fallback.route('**/*', (route) =>
        new URL(route.request().url()).origin === origin ? route.continue() : route.abort());
      if (disabled === 'dialog') await fallback.addInitScript(() => { HTMLDialogElement.prototype.showModal = undefined; });
      const fallbackPage = await fallback.newPage();
      for (const pathname of ['copilot/', 'copilot-dashboards/', 'frontier-analytics/']) {
        await fallbackPage.goto(base + pathname, { waitUntil: 'load' });
        assert.equal(await fallbackPage.locator('[data-dashboard-details] [data-dashboard-panel]').count(), 3);
        await fallbackPage.locator('[data-dashboard-open]').first().click();
        assert.equal(await fallbackPage.locator('#dashboard-business-leaders-ai-usage').isVisible(), true);
        assert.equal(await fallbackPage.evaluate(() => document.documentElement.scrollWidth <= innerWidth), true);
      }
      await fallback.close();
    }
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
