(in-ns 'example)

(def message
  (fn* []
    :hello/world))

(def ns (.deref *ns*))

(def file (.deref *file*))
