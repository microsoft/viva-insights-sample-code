---
layout: default
title: "Search"
permalink: /search/
description: "Search the Viva Insights Sample Code Library — find sample scripts, tutorials, and AI prompt examples across R, Python, Copilot, network, and causal-inference content."
search_exclude: true
---

# Search

{% include custom-navigation.html %}

<div class="vi-search">
  <input
    type="search"
    id="search-input"
    class="vi-search-input"
    placeholder="Search the library…"
    autocomplete="off"
    aria-label="Search the library"
    aria-controls="search-results"
  >
  <p id="search-status" class="vi-search-status" aria-live="polite"></p>
  <ul id="search-results" class="vi-search-results"></ul>
</div>

<script>window.SEARCH_INDEX_URL = "{{ '/search.json' | relative_url }}";</script>
<script src="https://cdn.jsdelivr.net/npm/lunr@2.3.9/lunr.min.js"></script>
<script src="{{ '/assets/js/search.js' | relative_url }}"></script>
