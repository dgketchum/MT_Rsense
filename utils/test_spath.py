# ===============================================================================
# Copyright 2017 dgketchum
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===============================================================================
import requests
from lxml import html

pageContent = requests.get('https://en.wikipedia.org/wiki/List_of_Olympic_medalists_in_judo')
tree = html.fromstring(pageContent.content)

goldWinners = tree.xpath('//*[@id="mw-content-text"]/table/tr/td[2]/a[1]/text()')

print goldWinners

# silverWinners=tree.xpath('//*[@id="mw-content-text"]/table/tr/td[3]/a[1]/text()')
# #bronzeWinner we need rows where there's no rowspan - note XPath
# bronzeWinners=tree.xpath('//*[@id="mw-content-text"]/table/tr/td[not(@rowspan=2)]/a[1]/text()')
# medalWinners=goldWinners+silverWinners+bronzeWinners
#
# medalTotals={}
# for name in medalWinners:
#     if medalTotals.has_key(name):
#         medalTotals[name]=medalTotals[name]+1
#     else:
#         medalTotals[name]=1
#
# for result in sorted(
#         medalTotals.items(), key=lambda x:x[1],reverse=True):
#         print '%s:%s' % result


# ===============================================================================
