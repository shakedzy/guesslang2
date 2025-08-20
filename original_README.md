# Guesslang [![Build Status](https://github.com/yoeo/guesslang/actions/workflows/python-package.yml/badge.svg)](https://github.com/yoeo/guesslang/actions) [![Documentation Status](https://readthedocs.org/projects/guesslang/badge/?version=latest)](http://guesslang.readthedocs.io/en/latest/?badge=latest) [![Pypi version](https://img.shields.io/pypi/v/guesslang.svg)](https://pypi.python.org/pypi/guesslang)

![Guesslang](docs/_static/images/guesslang-readme.png)

Guesslang detects the programming language of a given source code:

```bash
echo '
package main
import "fmt"

func main() {
    fmt.Println("My mascot is a gopher and Google loves me. Who am I?")
}

' | guesslang

# ⟶ Programming language: Go
```

Guesslang supports `54 programming languages`:

| Languages  |                |               |                |              |
|------------|----------------|---------------|----------------|--------------|
| `Assembly` | `Batchfile`    | `C`           | `C#`           | `C++`        |
| `Clojure`  | `CMake`        | `COBOL`       | `CoffeeScript` | `CSS`        |
| `CSV`      | `Dart`         | `DM`          | `Dockerfile`   | `Elixir`     |
| `Erlang`   | `Fortran`      | `Go`          | `Groovy`       | `Haskell`    |
| `HTML`     | `INI`          | `Java`        | `JavaScript`   | `JSON`       |
| `Julia`    | `Kotlin`       | `Lisp`        | `Lua`          | `Makefile`   |
| `Markdown` | `Matlab`       | `Objective-C` | `OCaml`        | `Pascal`     |
| `Perl`     | `PHP`          | `PowerShell`  | `Prolog`       | `Python`     |
| `R`        | `Ruby`         | `Rust`        | `Scala`        | `Shell`      |
| `SQL`      | `Swift`        | `TeX`         | `TOML`         | `TypeScript` |
| `Verilog`  | `Visual Basic` | `XML`         | `YAML`         |              |

With a guessing **accuracy higher than 90%**.

## Apps powered by Guesslang

#### Microsoft Visual Studio Code, automatic language detection

[Visual Studio Code](https://code.visualstudio.com/)
detects the programming language of the source code
that you paste into the editor
[using Guesslang machine learning model](https://code.visualstudio.com/updates/v1_60#_automatic-language-detection).

![](docs/_static/images/vscode.gif)

#### Chameledit

[Chameledit](https://github.com/yoeo/chameledit) is a simple web-editor
that automatically highlights your code.

![](docs/_static/images/chameledit.gif)

##### Other projects...

* [Pasta](https://github.com/yoeo/pasta), the [Slack](https://slack.com)
  bot that pretty-pastes source code.
* [GG](https://github.com/yoeo/gg), a silly guessing game.

## Documentation

* Guesslang documentation is available at
  https://guesslang.readthedocs.io/en/latest/

* Guesslang language detection explained here
  https://guesslang.readthedocs.io/en/latest/contents.html#how-does-guesslang-guess

* Guesslang is based on [Tensorflow](https://github.com/tensorflow/tensorflow)
  machine learning framework

## Installation

* Python 3.7+ is required

* Install the latest stable version:

```bash
pip3 install guesslang
```

* or install Guesslang from source code:

```bash
pip3 install .
```

* Windows specific

To run [Tensorflow](https://github.com/tensorflow/tensorflow)
on Microsoft Windows you need to install Visual C++ runtime libraries,
available on [Microsoft website](https://www.microsoft.com/en-us/download/details.aspx?id=53587)

## Guesslang command line

* Show all available options

```bash
guesslang --help
```

* Detect the programming language of ``/etc/bashrc`` configuration file:

```bash
guesslang /etc/bashrc

# ⟶ Programming language: Shell
```

* Detect the programming language of a given text:

```bash
echo '
/** Turn command line arguments to uppercase */
object Main {
  def main(args: Array[String]) {
    val res = for (a <- args) yield a.toUpperCase
    println("Arguments: " + res.toString)
  }
}
' | guesslang

# ⟶ Programming language: Scala
```

* Show the detection probabilities for a given source code:

```bash
echo "
def qsort(items):
    if not items:
        return []
    else:
        pivot = items[0]
        less = [x for x in items if x <  pivot]
        more = [x for x in items[1:] if x >= pivot]
        return qsort(less) + [pivot] + qsort(more)


if __name__ == '__main__':
    items = [1, 4, 2, 7, 9, 3]
    print(f'Sorted: {qsort(items)}')

" | guesslang --probabilities

# Language name       Probability
#  Python               74.80%
#  Haskell               6.73%
#  CoffeeScript          5.32%
#  Groovy                1.95%
#  Markdown              0.93%
#  ...
```

## Guesslang Python package

* Guesslang can be used as a Python package.
  [Package documentation available here](https://guesslang.readthedocs.io/en/latest/guesslang.html)

``` python

from guesslang import Guess


guess = Guess()

name = guess.language_name("""
    % Quick sort

    -module (recursion).
    -export ([qsort/1]).

    qsort([]) -> [];
    qsort([Pivot|T]) ->
          qsort([X || X <- T, X < Pivot])
          ++ [Pivot] ++
          qsort([X || X <- T, X >= Pivot]).
""")

print(name)  # ⟶ Erlang
```

## License and credits

* [Guesslang documentation](https://guesslang.readthedocs.io/en/latest/)

* Gesslang training dataset created with
  [GuesslangTools](https://github.com/yoeo/guesslangtools)

* Guesslang developped with [Tensorflow](https://www.tensorflow.org/)

* Guesslang icon created with
  [AndroidAssetStudio](https://github.com/romannurik/AndroidAssetStudio)
  and
  [Eduardo Tunni's Warnes font](https://fonts.google.com/specimen/Warnes)

* Example source codes used here retrieved from
  [Rosetta Code](https://rosettacode.org/wiki/Sorting_algorithms/Quicksort)

* Guesslang — Copyright (c) 2021 Y. SOMDA, [MIT License](LICENSE)
