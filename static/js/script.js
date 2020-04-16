window.addEventListener("load", (event) => {
  // Get HTML image
  const image = document.querySelector("#generated-image");
  // Get HTML button
  const button = document.querySelector("button");
  button.onclick = (event) => {
    fetch(`${window.origin}/generate`)
      .then(response => response.json())
      .then(data => {
        image.src = "data:image/png;base64, " + data.image
      })
  }
});
