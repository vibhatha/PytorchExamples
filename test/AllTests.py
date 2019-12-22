from unittest import TestCase
from util import LogSaver


class AllTests(TestCase):

    def test1(self):
        print("Test")
        LogSaver.save_log("test1", stat="sss")
