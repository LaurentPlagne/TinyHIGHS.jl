import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';

mermaid.initialize({
    startOnLoad: false,
    theme: 'neutral',
    securityLevel: 'loose'
});

window.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll("pre code.language-mermaid").forEach((codeBlock) => {
        const pre = codeBlock.parentElement;
        const div = document.createElement("div");
        div.className = "mermaid";
        div.textContent = codeBlock.textContent;
        pre.replaceWith(div);
    });
    mermaid.run();
});
