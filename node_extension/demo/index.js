var app = require('express')();
var http = require('http').Server(app);
var io = require('socket.io')(http);

var sampler = require("../build/Release/samplerproxy");
var samplerProxy = new sampler.SamplerProxy();

function OnConnection() {
  samplerProxy.setupExperiment();
  samplerProxy.reset();
}

function OnSample() {
  var num_iterations = 10000;
  samplerProxy.reset();
  var result_histogram_str = samplerProxy.testMetroInfer(num_iterations);
  io.emit('histogram message', ResultToGraphData(result_histogram_str));
}

function ResultToGraphData(result_str) {
  var data = {
    labels: [],
    datasets: [
      {
        label: 'Canned',
        fillColor: "rgba(220,220,220,0.2)",
        strokeColor: "rgba(220,220,220,1)",
        pointColor: "rgba(220,220,220,1)",
        pointStrokeColor: "#fff",
        pointHighlightFill: "#fff",
        pointHighlightStroke: "rgba(220,220,220,1)",
        data: []
      }]
  };

  var result = JSON.parse(result_str);
  data.labels = result.values.slice();
  data.datasets[0].data = result.data.slice();
  return data;
}

app.get('/', function(req, res){
  res.sendFile(__dirname + '/index.html');
});

io.on('connection', function(socket){
  OnConnection();
  socket.on('sample message', function(msg){
    OnSample()
  });
});

http.listen(3000, function(){
  console.log('listening on *:3000');
});
