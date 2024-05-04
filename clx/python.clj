(in-ns python)

(def builtins* (import* builtins))

(defn make-binding [name]
  `(def ~(symbol name) (python* ~name)))

(defn dosnt-start-with-underscore [x]
  (not (python* x ".startswith('_')")))

(defmacro define-bindings []
  `(do ~@(map make-binding
           (filter dosnt-start-with-underscore
             (builtins/dir builtins*)))))

(define-bindings)
