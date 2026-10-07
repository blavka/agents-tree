#!/usr/bin/env bash
# Create the throwaway "Acme Bookshop" demo project agents-tree is checked and
# photographed against. Usage: docs/demo/setup.sh [DIR]   (default /tmp/agents-tree-demo)
#
# The project is small but gives agents real work: three planted bugs, a TODO,
# almost no tests. It is its own git repository with a made-up author, so
# nothing from the machine it runs on ends up in it.
set -euo pipefail

dir="${1:-/tmp/agents-tree-demo}"
# One atomic mkdir, private to this user: it fails if anything (a directory, a
# symlink) is already there, so nobody else on a shared /tmp can slip a planted
# directory under the code this script writes and the agents later run.
if ! mkdir -m 700 "$dir"; then
  echo "could not create $dir; if it is left from an earlier demo, remove it first" \
       "(see docs/demo/README.md, Clean up)" >&2
  exit 1
fi
cd "$dir"
mkdir bookshop tests

cat > README.md <<'EOF'
# Acme Bookshop

A tiny bookshop backend used as a playground: a catalog, a cart, and an inventory
that reserves stock at checkout.

```bash
python -m bookshop
```

## Known issues

- Discounts are hard-coded in the cart.
- Inventory reservations are never released when a checkout fails.
- There are hardly any tests.
EOF

cat > bookshop/__init__.py <<'EOF'
"""Acme Bookshop: catalog, cart and inventory."""
EOF

cat > bookshop/catalog.py <<'EOF'
from dataclasses import dataclass


@dataclass(frozen=True)
class Book:
    isbn: str
    title: str
    author: str
    price_cents: int


BOOKS = {
    "978-0-13-468599-1": Book("978-0-13-468599-1", "The Pragmatic Programmer", "Hunt, Thomas", 4299),
    "978-0-201-63361-0": Book("978-0-201-63361-0", "Design Patterns", "Gamma et al.", 5499),
    "978-1-59327-584-6": Book("978-1-59327-584-6", "The Linux Command Line", "Shotts", 3495),
    "978-0-262-03384-8": Book("978-0-262-03384-8", "Introduction to Algorithms", "Cormen et al.", 8999),
}


def find(isbn):
    return BOOKS.get(isbn)


def search(text):
    text = text.lower()
    return [b for b in BOOKS.values() if text in b.title.lower() or text in b.author.lower()]
EOF

cat > bookshop/cart.py <<'EOF'
from bookshop import catalog
from bookshop.inventory import Inventory


class Cart:
    def __init__(self, inventory: Inventory):
        self.inventory = inventory
        self.items = {}  # isbn -> quantity

    def add(self, isbn, quantity=1):
        if catalog.find(isbn) is None:
            raise KeyError(isbn)
        self.items[isbn] = self.items.get(isbn, 0) + quantity

    def remove(self, isbn):
        del self.items[isbn]

    def subtotal_cents(self):
        return sum(catalog.find(isbn).price_cents * qty for isbn, qty in self.items.items())

    def total_cents(self, code=None):
        total = self.subtotal_cents()
        # TODO: discounts should come from configuration, not code
        if code == "WELCOME10":
            total = total * 0.9
        elif code == "BOOKWORM":
            total = total - 500
        return total  # BUG: may be a float, and may go below zero

    def checkout(self):
        for isbn, qty in self.items.items():
            self.inventory.reserve(isbn, qty)  # BUG: earlier reservations leak if this raises
        order = dict(self.items)
        self.items.clear()
        return order
EOF

cat > bookshop/inventory.py <<'EOF'
class OutOfStock(Exception):
    pass


class Inventory:
    def __init__(self, stock=None):
        self.stock = dict(stock or {})
        self.reserved = {}

    def available(self, isbn):
        return self.stock.get(isbn, 0) - self.reserved.get(isbn, 0)

    def reserve(self, isbn, quantity):
        if self.available(isbn) < quantity:
            raise OutOfStock(isbn)
        self.reserved[isbn] = self.reserved.get(isbn, 0) + quantity

    def release(self, isbn, quantity):
        self.reserved[isbn] = self.reserved.get(isbn, 0) - quantity  # BUG: can go negative

    def ship(self, isbn, quantity):
        self.stock[isbn] -= quantity
        self.reserved[isbn] -= quantity
EOF

cat > bookshop/__main__.py <<'EOF'
from bookshop.cart import Cart
from bookshop.inventory import Inventory

inventory = Inventory({"978-0-13-468599-1": 3, "978-1-59327-584-6": 1})
cart = Cart(inventory)
cart.add("978-0-13-468599-1", 2)
cart.add("978-1-59327-584-6")
print("total:", cart.total_cents("WELCOME10") / 100)
print("order:", cart.checkout())
EOF

cat > tests/test_catalog.py <<'EOF'
from bookshop import catalog


def test_search_by_author():
    assert [b.title for b in catalog.search("shotts")] == ["The Linux Command Line"]
EOF

printf '__pycache__/\n.pytest_cache/\n' > .gitignore

git init -q
git -c user.name=Demo -c user.email=demo@example.com -c commit.gpgsign=false add -A
git -c user.name=Demo -c user.email=demo@example.com -c commit.gpgsign=false \
  commit -q -m "Acme Bookshop playground"

python3 -m bookshop >/dev/null
echo "demo project ready in $dir"
