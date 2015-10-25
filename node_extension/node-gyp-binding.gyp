{
  "targets": [
    {
      "target_name": "samplerproxy",
      "sources": [ "src/node_sampler_proxy.cpp" ],
      "include_dirs": [
        # Path to hopper root
      ],
      "libraries": [
        # Path to hopper framework library
        # Path to hopper histogram library
      ],
      "cflags" : [ "-std=c++11", "-stdlib=libc++" ],
      "conditions": [
        [ 'OS!="win"', {
          "cflags+": [ "-std=c++11" ],
          "cflags_c+": [ "-std=c++11" ],
          "cflags_cc+": [ "-std=c++11" ],
        }],
        [ 'OS=="mac"', {
          "xcode_settings": {
            "OTHER_CPLUSPLUSFLAGS" : [ "-std=c++11", "-stdlib=libc++" ],
            "OTHER_LDFLAGS": [ "-stdlib=libc++" ],
            "MACOSX_DEPLOYMENT_TARGET": "10.7"
          },
        }],
        ],
    }
  ]
}
