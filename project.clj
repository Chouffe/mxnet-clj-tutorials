(defproject mxnet-clj-tutorials "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.10.0"]
                 [org.apache.mxnet.contrib.clojure/clojure-mxnet-linux-cpu "1.4.0"]

                 ;; OpenCV wrapper
                 [origami "4.0.0-7"]]
  :plugins [[lein-jupyter "0.1.16"]]

  :repositories
  [["vendredi" {:url "https://repository.hellonico.info/repository/hellonico/"}]
   ["staging" {:url "https://repository.apache.org/content/repositories/staging"
               :snapshots true
               :update :always}]
   ["snapshots" {:url "https://repository.apache.org/content/repositories/snapshots"
                 :snapshots true
                 :update :always}]])
