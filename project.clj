(defproject mxnet-clj-tutorials "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.10.0"]
                 [org.apache.mxnet.contrib.clojure/clojure-mxnet "1.5.0-SNAPSHOT"]
                 ; [org.apache.mxnet.contrib.clojure/clojure-mxnet-linux-gpu "1.4.1"]

                 ;; OpenCV wrapper
                 [origami "4.0.0-7"]]
  ;; Jupyter Notebook plugin
  :plugins [[lein-jupyter "0.1.16"]]

  :repositories
  [["vendredi" {:url "https://repository.hellonico.info/repository/hellonico/"}]
   ["staging" {:url "https://repository.apache.org/content/repositories/staging"
               :snapshots true
               :update :always}]
   ["snapshots" {:url "https://repository.apache.org/content/repositories/snapshots"
                 :snapshots true
                 :update :always}]])
