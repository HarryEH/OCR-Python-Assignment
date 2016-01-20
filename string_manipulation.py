
def chars_to_words(spaces, chars):
    """ Takes a list of chars and returns a list of words based on 1 or 0 in the spaces list.

    :param spaces: this is the list that contains zeros or ones
    :param chars: this is a list of characters
    :return: list of strings that are split up based on whether there is a one or a zero.
    """
    y = 1
    i_index = 0
    for i in spaces:
        if float(i) == 1.:
            chars.insert(i_index+y, ' ')
            y += 1
        i_index += 1

    return chars


def error_correction(all_words, word_tester):
    """Extremely simple error correction, takes the mistakes and replaces letters in thenm
    and chooses the most popular correct word from the list of replacments.
    :rtype: list
    :param all_words is the list of words that the words are checked against
    :param word_tester to compare against the list of all words"""

    alpha = 'abcdefghijklmnopqrstuvwxyz'
    lis_with_count = []
    for line in file('words_counts.txt'):
        lis_with_count.append(line.split('\t'))

    # These are the names from the jungle book - taken from kiplingsociety.co.uk
    exceptions = ['waingunga', 'tabaqui', 'shere', 'lungri', 'mowgli', 'akela', 'baloo', 'raksha', 'bagheera','ikki',
                  'mao', 'kaa', 'mang', 'hathi', 'chil', 'mohwa', 'tha', 'mysa', 'messua', 'khanhiwara', 'buldeo',
                  'purun', 'dass', 'rama', 'nilghai', 'thuu', 'phao', 'pheeal', 'dhole', 'won', 'tolla', 'dekkan',
                  'lahini', 'ferao', 'rikki', 'tikki', 'tavi', 'darzee', 'karait', 'chuchundra', 'novastoshnah',
                  'matkah', 'kotick','vitch', 'magellan', 'bhagat', 'chota', 'simla', 'kali', 'langurs', 'bara',' singh',
                  'mugger', 'ghaut','gavial', 'mohoo', 'chapta', 'batchua', 'chilwa', 'kikar', 'quiqern', 'hira',
                  'guj']

    mistake_lis = [word for word in word_tester if not((word[0].lower()+word[1:]) in exceptions) and
                   not(binary_search(all_words, (word[0].lower() + word[1:])))]

    edit_str = ''
    test_item = ' '
    for w in mistake_lis:

        w_lower = w.lower()

        same_len_lis = [x for x in all_words if len(x) == len(w)]  # List Comp so that the list of all words is only
        # words of the same length as the word being tested - bcs of nature of classifiers.

        split_word = [(w_lower[:i], w_lower[i:]) for i in range(len(w) + 1)]

        replace_letters = [p1 + p3 + p2[1:] for p1, p2 in split_word for p3 in alpha if p2]

        real_words = [word for word in replace_letters if binary_search(same_len_lis, word)]

        maxN = 0
        for ea_word in real_words:
            lis_with_counts_sorted = [item for item in lis_with_count if item[0] == ea_word]
            if lis_with_counts_sorted:
                for w1, count in lis_with_counts_sorted:
                    if int(count) > maxN:
                        maxN = int(count)
                        edit_str = w1
                        test_item = w1

        # GIVES THE WORD A CAPITAL IF IT STARTED WITH ONE IN THE FIRST PLACE - this could lead to errors but it
        # is safer than just giving everything as lowercase.
        if real_words:
            if w[0].isupper() and edit_str == test_item and len(w) == len(edit_str):
                word_tester[word_tester.index(w)] = edit_str[0].upper() + edit_str[1:]
            elif edit_str == test_item and len(w) == len(edit_str):
                word_tester[word_tester.index(w)] = edit_str
    return word_tester


def binary_search(lis, item):
    """Binary search alfgorithm that should run faster over large data sets than a linear search.

    :param lis: list of items to search
    :param item: item that you search for
    :return: a boolean
    """
    fst = 0
    lst = len(lis)-1
    while fst <= lst:
        mid = (fst + lst)//2
        if lis[mid] == item:
            return True
        else:
            if item < lis[mid]:
                lst = mid-1
            else:
                fst = mid+1
    return False




