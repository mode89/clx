(in-ns 'compiler-test)

(load "clx/compiler")
(alias 'comp 'clx.compiler)

(load "clx/test")
(refer* 'clx.test 'deftest)
(refer* 'clx.test 'is)
(refer* 'clx.test 'raises)

(deftest read-string
  (is (= (comp/read-string "1") 1))
  (is (= (comp/read-string "1.23") 1.23))
  ; TODO (is (= (comp/read-string "1.23e4") 1.23e4))
  (is (= (comp/read-string "true") true))
  (is (= (comp/read-string "false") false))
  (is (= (comp/read-string "nil") nil))
  (is (= (comp/read-string "\"hello world\"") "hello world"))
  (is (= (comp/read-string "\"a \\\\foo\\nbar\\\" b\"") "a \\foo\nbar\" b"))
  (raises Exception "Unterminated string"
    (comp/read-string "\"hello world"))
  (is (= (comp/read-string ":hello") :hello))
  (is (= (comp/read-string ":hello/world") :hello/world))
  (is (= (comp/read-string "hello") 'hello))
  (is (= (comp/read-string "hello/world") 'hello/world)))
