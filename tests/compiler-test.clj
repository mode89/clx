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
  (raises Exception #"Unterminated string"
    (comp/read-string "\"hello world"))
  (is (= (comp/read-string ":hello") :hello))
  (is (= (comp/read-string ":hello/world") :hello/world))
  (is (= (comp/read-string "hello") 'hello))
  (is (= (comp/read-string "hello/world") 'hello/world))
  (is (identical? (comp/read-string "()") '()))
  (is (list? (comp/read-string "(1 2 3)")))
  (is (= (comp/read-string "(1 2 3)") '(1 2 3)))
  (is (= (comp/read-string "(1 (2 3) ((4 5) 6))") '(1 (2 3) ((4 5) 6))))
  (raises Exception #"Expected '\)'"
    (comp/read-string "(1 2 3"))
  (is (vector? (comp/read-string "[4 5 6]")))
  (is (= (comp/read-string "[4 5 6]") [4 5 6]))
  (is (= (comp/read-string "[4 [5 6] [7 [8 9]]]") [4 [5 6] [7 [8 9]]]))
  (raises Exception #"Expected ']'"
    (comp/read-string "[4 5 6"))
  (is (map? (comp/read-string "{:a 7 \"b\" eight}")))
  (is (= (comp/read-string "{:a 7 \"b\" eight}") {:a 7 "b" 'eight}))
  (raises Exception #"Expected '}'"
    (comp/read-string "{1 2 3 4"))
  (is (= (comp/read-string "'hello") '(quote hello))))

(deftest munge
  (is (= (comp/munge "foo") "foo"))
  (is (= (comp/munge "foo.bar/baz") "foo_DOT_bar_SLASH_baz"))
  (is (= (comp/munge "foo-bar.*baz*/+qux_fred!")
         "foo_DASH_bar_DOT__STAR_baz_STAR__SLASH__PLUS_qux_fred_BANG_"))
  (is (= (comp/munge "if") "if_"))
  (is (= (comp/munge "def") "def_"))
  (raises Exception #"reserved"
    (comp/munge "def_")))

; (deftest def
;   (eval))
