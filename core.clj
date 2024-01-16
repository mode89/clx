(def fn ^{:macro true}
  (fn* margs
    (let* [spec (first margs)
           params# (gensym "params__")
           bindings (destructure1 spec params#)
           body (rest margs)]
      `(fn* ~params#
        (let* ~bindings
          (do ~@body))))))

(def defn ^{:macro true}
  (fn [name params & body]
    `(def ~name
       (fn ~params ~@body))))

(def defmacro ^{:macro true}
  (fn [name params & body]
    `(def ~name ^{:macro true}
       (fn ~params ~@body))))

(defmacro let [bindings & body]
  `(let* ~bindings
     ~@body))

(defmacro if-not [test then & else]
  `(if (not ~test)
     ~then
     ~@else))

(defmacro when [test & body]
  `(if ~test
     (do ~@body)
     nil))

(defmacro when-not [test & body]
  `(if-not ~test
     (do ~@body)
     nil))

(defmacro assert [expr & args]
  (let [message (if (= (count args) 0)
                  "Assertion failed"
                  (if (= (count args) 1)
                    (str "Assertion failed: " (first args))
                    (throw (str "assert takes 1 or 2 arguments, got "
                                (+ (count args) 1) " instead"))))]
    `(when-not ~expr
       (throw ~message))))
