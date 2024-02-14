(in-ns clx.core)

(def throw
  (fn* clx.core/throw [x]
    (python* "raise " x)))

(def Exception
  (python* "Exception"))

(def if ^{:macro? true}
  (fn* clx.core/if [pred then else]
    `(cond ~pred ~then true ~else)))

(def when ^{:macro? true}
  (fn* clx.core/when [pred & body]
    `(cond ~pred (do ~@body))))

(def when-not ^{:macro? true}
  (fn* clx.core/when-not [pred & body]
    `(cond
       ~pred nil
       true (do ~@body))))

(def fn ^{:macro? true}
  (fn* clx.core/fn [& args]
    (let* [arg1 (first args)]
      (if (symbol? arg1)
        `(fn* ~arg1 ~(second args)
          (do ~@(rest (rest args))))
        `(fn* ~arg1
          (do ~@(rest args)))))))

(def defn ^{:macro? true}
  (fn* clx.core/defn [name params & body]
    `(def ~name
      (with-meta
        (fn ~name ~params ~@body)
        ~(meta name)))))

(def defmacro ^{:macro? true}
  (fn* clx.core/defmacro [name params & body]
    `(defn ~(with-meta name {:macro? true}) ~params
      (do ~@body))))

(defmacro let [bindings & body]
  `(let* ~bindings
    (do ~@body)))

(defn name [x]
  (python* x ".name"))

(defmacro set! [obj field value]
  `(let [obj# ~obj
         value# ~value]
     (python* obj# "." ~(name field) " = " value#)))

(defmacro lazy-seq [& body]
  `(lazy-seq* (fn [] ~@body)))
