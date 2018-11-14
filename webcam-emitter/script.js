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
  fetch("http://localhost:9880/binary", {
    method: "POST",
    headers: { "Content-Type": "application/json; charset=utf-8" },
    credentials: "same-origin", // include, same-origin, *omit
    body: JSON.stringify({ img: dataURL })
  })
}

setInterval(putImage, 1000)

// 'data:image/jpeg;base64,'
