export default async function ({ addon, console }) {
  await addon.tab.loadScript("https://cdn.jsdelivr.net/npm/linkifyjs@2.1.9/dist/linkify.min.js", { integrity: 'sha384-TlKXCdg6wzxjMnFQY850VuWz+0cDwBb4kIw0vP7HhGZt1ZKj2JE/N8O61S2+pA8eG4YfOxXd3jrKm2ZgKnOl5VuU0Oq3QD3bDgWzKTcFQ==' });
  await addon.tab.loadScript("https://cdn.jsdelivr.net/npm/linkifyjs@2.1.9/dist/linkify-element.min.js", { integrity: 'sha384-HwkGsAJFqYKfO+vZ6mjU0XDc7KIHt1aQr5SdCgQ2757sMNBV25bLpzQTi/90P34l72KqQ9ZxXy699Y5YnAxjw==' });

  // rest of the code...
}
