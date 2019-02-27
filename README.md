# mxnet-clj-tutorials

A Collection of tutorials for the Clojure MXNET package

## How to run the examples

* A Dockerfile is provided for you to build locally: [https://github.com/Chouffe/mxnet-clj](Dockerfile)
* Run the following docker command to start a REPL
```
docker run \
       --rm \
       --env="DISPLAY" \
       --env="QT_X11_NO_MITSHM=1" \
       --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
       --volume "$PWD:/home/mxnetuser/app" \
       --volume "$HOME/.m2:/home/mxnetuser/.m2" \
       --volume "$HOME/.m2:/root/.m2" \
       --interactive \
       -p "12121:12121" \
       --tty \
       chouffe/mxnet-clj-cpu \
       lein repl :start :host 0.0.0.0 :port 12121
```

## License

Copyright © 2019 FIXME

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
