export default async function ({ addon, console }) {
  await addon.tab.loadScript("https://cdn.jsdelivr.net/npm/linkifyjs@2.1.9/dist/linkify.min.js");
  await addon.tab.loadScript("https://cdn.jsdelivr.net/npm/linkifyjs@2.1.9/dist/linkify-element.min.js");

  const pageType = document.location.pathname.substr(1).split("/")[0];

  switch (pageType) {
    case "users":
      linkifyElement(document.querySelectorAll("#user-details .read-only"));
      break;

    case "projects":
      linkifyElement(document.querySelector(".project-description"));
      break;

    case "studios":
      linkifyElement(document.querySelector("#description.read-only .overview"));
      break;
  }

  (async () => {
    while (true) {
      let comment = await addon.tab.waitForElement(".comment", { markAsSeen: true });
      linkifyElement(comment);
    }
  })();
}
