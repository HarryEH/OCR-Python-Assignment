import bulk_code as bc

bc.main()


# THIS CODE WAS WRITTEN SO THAT I COULD USE words_counts.txt
# THIS WAS RUN ONCE TO CREATE A NEW FILE
# with open('wordsEn.txt', 'r') as f:
#         all_words = f.read().replace('\r\n', ' ')
# all_words = all_words.split()
#
# # print all_words
# lis_with_count = []
# for line in file('count_1w.txt'):
#     lis = line.split('\t')
#     # print lis[0]
#     if sm.binary_search(all_words, lis[0]):
#         # print lis[0]
#         lis_with_count.append(lis[0]+'\t'+lis[1][:-2])
#
#
# with open('words_counts.txt', 'w') as w:
#     w.writelines('\n'.join(str(l) for l in lis_with_count))



