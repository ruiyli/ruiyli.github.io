document.addEventListener("DOMContentLoaded", function () {
  const wechatModal = document.getElementById("WeChatMod");
  if (!wechatModal) return;

  document.querySelectorAll("#WeChatBtn").forEach(function (wechatBtn) {
    wechatBtn.addEventListener("click", function (event) {
      event.preventDefault();
      wechatModal.style.display = "block";
    });
  });

  wechatModal.addEventListener("click", function (event) {
    if (event.target === wechatModal) {
      wechatModal.style.display = "none";
    }
  });

  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape") {
      wechatModal.style.display = "none";
    }
  });
});
