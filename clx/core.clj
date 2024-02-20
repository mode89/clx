(in-ns clx.core)

(import* operator)

(def throw
  (fn* clx.core/throw [x]
    (python* "raise " x)))

(def Exception
  (python* "Exception"))

(def if ^{:macro? true}
  (fn* clx.core/if [pred then else]
    `(cond
       ~pred ~then
       true ~else)))

(def if-not ^{:macro? true}
  (fn* clx.core/if-not [pred then else]
    `(cond
       ~pred ~else
       true ~then)))

(def when ^{:macro? true}
  (fn* clx.core/when [pred & body]
    `(cond ~pred (do ~@body))))

(def when-not ^{:macro? true}
  (fn* clx.core/when-not [pred & body]
    `(cond
       ~pred nil
       true (do ~@body))))

(def assert ^{:macro? true}
  (fn* clx.core/assert [tst & msg]
    `(when-not ~tst
       (throw (Exception ~@msg)))))

(def multi-arity-fn
  (fn* multi-arity-fn [fname decls]
    (let* [variadic? (fn* [args]
                       (loop* [args* args]
                         (when (seq args*)
                           (if (operator/eq '& (first args*))
                             true
                             (recur (rest args*))))))
           arity-name (fn* [arity]
                        (when fname
                          (vector
                            (symbol
                              (let* [fname* (python* "str(" fname ")")
                                     arity* (python* "str(" arity ")")]
                                (python*
                                  fname*
                                  " + \"-arity-\" + "
                                  arity*))))))
           decl-entry (fn* [decl]
                        (let* [args (first decl)
                               _ (assert (vector? args)
                                   "args must be a vector")
                               body (rest decl)
                               arity (if (variadic? args)
                                       :variadic
                                       (count args))]
                          `(list ~arity
                             (fn* ~@(arity-name arity) [~@args]
                               (do ~@body)))))]
      `(let* [arities# (list ~@(loop* [decls* decls
                                       entries (list)]
                                 (if (seq decls*)
                                   (recur
                                     (rest decls*)
                                     (cons (decl-entry (first decls*))
                                           entries))
                                   entries)))
              arities-dict# (python* "dict(" arities# ")")
              variadic# (.get arities-dict# :variadic)]
         (fn* ~@(when fname [fname]) [& args]
           (python*
             "f = " arities-dict# ".get(len(" args "), " variadic# ")\n"
             "if f is None:\n"
             "    raise Exception(\"Wrong number of arguments\")\n"
             "f(*" args ")"))))))

(def fn ^{:macro? true}
  (fn* clx.core/fn [& args]
    (if (symbol? (first args))
      (let* [fname (first args)]
        (if (vector? (second args))
          (let* [params (second args)]
            `(fn* ~fname ~params
               (do ~@(rest (rest args)))))
          (multi-arity-fn fname (rest args))))
      (if (vector? (first args))
        (let* [params (first args)]
          `(fn* ~params
             (do ~@(rest args))))
        (multi-arity-fn (gensym "___fn_") args)))))

(def defn ^{:macro? true}
  (fn* clx.core/defn [name & decl]
    `(def ~name
      (with-meta
        (fn ~name ~@decl)
        ~(meta name)))))

(def defmacro ^{:macro? true}
  (fn* clx.core/defmacro [name & decl]
    `(defn ~(with-meta name {:macro? true}) ~@decl)))

(defn even? [x]
  (python* "not " x " & 1"))

(defn odd? [x]
  (python* x " & 1 == 1"))

(defn inc [x]
  (python* x " + 1"))

(defn dec [x]
  (python* x " - 1"))

(def + operator/add)
(def - operator/sub)
(def * operator/mul)
(def / operator/truediv)
(def = operator/eq)
(def not= operator/ne)
(def < operator/lt)
(def > operator/gt)
(def <= operator/le)
(def >= operator/ge)

(defmacro let [bindings & body]
  (assert (vector? bindings) "bindings must be a vector")
  (assert (even? (count bindings))
    "bindings must have an even number of elements")
  `(let* ~bindings
    (do ~@body)))

(defmacro when-let [bindings & body]
  (assert (vector? bindings) "bindings must be a vector")
  (assert (operator/eq 2 (count bindings))
    "bindings must have exactly two elements")
  (let [bname (bindings 0)
        bvalue (bindings 1)]
    `(let [~bname ~bvalue]
       (when ~bname ~@body))))

(defn name [x]
  (python* x ".name"))

(defmacro set! [obj field value]
  `(let [obj# ~obj
         value# ~value]
     (python* obj# "." ~(name field) " = " value#)))

(defmacro lazy-seq [& body]
  `(lazy-seq* (fn [] ~@body)))
