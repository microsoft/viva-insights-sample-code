---
layout: default
title: "Search"
permalink: /search/
description: "Search the Viva Insights Sample Code Library — find sample scripts, tutorials, and AI prompt examples across R, Python, Copilot, network, and causal-inference content."
search_exclude: true
---

# Search

{% include custom-navigation.html %}

<p class="vi-search-lede">Find sample scripts, tutorials, and prompt examples across R, Python, Copilot, network analysis, and causal inference. Press <kbd>/</kbd> from anywhere on this page to focus the search box.</p>

<div class="vi-search">
  <div class="vi-search-field">
    <svg width="18" height="18" viewBox="0 0 16 16" fill="currentColor" aria-hidden="true" focusable="false"><path d="M11.742 10.344a6.5 6.5 0 1 0-1.397 1.398h-.001q.044.06.098.115l3.85 3.85a1 1 0 0 0 1.415-1.414l-3.85-3.85a1 1 0 0 0-.115-.1zM12 6.5a5.5 5.5 0 1 1-11 0 5.5 5.5 0 0 1 11 0"/></svg>
    <input
      type="search"
      id="search-input"
      class="vi-search-input"
      placeholder="Search the library…"
      autocomplete="off"
      aria-label="Search the library"
      aria-controls="search-results"
    >
    <span class="vi-search-kbd" aria-hidden="true">/</span>
  </div>
  <p id="search-status" class="vi-search-status" aria-live="polite"></p>
  <ul id="search-results" class="vi-search-results"></ul>
</div>

<script>
  window.SEARCH_INDEX_URL = "{{ '/search.json' | relative_url }}";
  // "/" focus shortcut, suppressed when typing in an input/textarea.
  document.addEventListener('keydown', function (e) {
    if (e.key !== '/' || e.metaKey || e.ctrlKey || e.altKey) return;
    var el = document.activeElement;
    if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable)) return;
    var input = document.getElementById('search-input');
    if (!input) return;
    e.preventDefault();
    input.focus();
    input.select();
  });
</script>
<script src="https://cdn.jsdelivr.net/npm/lunr@2.3.9/lunr.min.js"></script>
<script src="{{ '/assets/js/search.js' | relative_url }}"></script>
