BOOK_PATHS = {
    "The Count of Monte Cristo": "data/books/The Count of Monte Cristo.txt",
    "In Search of the Castaways": "data/books/In search of the castaways.txt",
}

def load_book(book_name):
    path = BOOK_PATHS[book_name]
    with open(path, encoding="utf-8") as f:
        return f.read()
