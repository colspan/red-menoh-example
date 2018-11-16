var constraints = { optional: [{ width: 640 }, { height: 480 }] }

function didntGetUserMedia(stream) {
  throw new Error("cannot get local media.")
}

function gotUserMedia(stream) {
  var video = document.getElementById("video")
  video.srcObject = stream
  video.play()
  var videoTrack = stream.getVideoTracks()[0]
  videoTrack.applyConstraints(constraints)
}
navigator.getUserMedia(
  { audio: false, video: true },
  gotUserMedia,
  didntGetUserMedia
)

function captureFrame() {
  var canvas = document.getElementById("video-buffer")
  var ctx = canvas.getContext("2d")
  var width = constraints.optional[0].width
  var height = constraints.optional[1].height
  canvas.width = width
  canvas.height = height
  ctx.drawImage(video, Math.floor((height - width) / 2), 0)
  var dataURL = canvas.toDataURL("image/jpeg")
  return dataURL
}

function putImage() {
  var dataURL = captureFrame()
  fetch("http://localhost:9880/vgg16", {
    method: "POST",
    headers: { "Content-Type": "application/json; charset=utf-8" },
    credentials: "same-origin", // include, same-origin, *omit
    body: JSON.stringify({ img: dataURL })
  })
}

setInterval(putImage, 1000)

var connection = new WebSocket('ws://localhost:8080');

var detectedDisplayElems = [];
connection.onopen = function () {
  var resultContainerElem = document.getElementById('detected-result-container');
  for(var i = 0; i<5; i++){
    var detectedCategoryElem = document.createElement('div');
    detectedCategoryElem.classList.add('detected-category');
    resultContainerElem.appendChild(detectedCategoryElem);

    var detectedScoreElem = document.createElement('div');
    detectedScoreElem.classList.add('detected-score');
    resultContainerElem.appendChild(detectedScoreElem);
    detectedDisplayElems.push({
      category: detectedCategoryElem,
      score: detectedScoreElem
    });
  }
};

connection.onmessage = function (e) {
  var data =JSON.parse(e.data)[1];
  for(var i=0;i<5;i++){
    ['category', 'score'].forEach(elem=>{
      detectedDisplayElems[i][elem].innerHTML = data[i][elem];
    })
  }
};
