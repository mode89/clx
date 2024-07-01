(in-ns 'clx.test)

(import* pytest)

(def raises* pytest/raises)

(defmacro deftest [tname & body]
  `(defn ~(symbol (str "test-" tname)) []
     ~@body))

(defmacro is [x]
  `(assert ~x "Test failed"))

(defmacro raises [ex-type arg0 & args]
  (if (re-pattern? arg0)
    `(let [pattern# ~arg0]
       (python/with [_ (apply-kw* clx.test/raises*
                         ~ex-type {"match" pattern#})]
         ~@args))
    `(python/with [_ (clx.test/raises* ~ex-type)]
       ~arg0
       ~@args)))
