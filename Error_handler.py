import sys
import os


class ErrorHandler:

    @staticmethod
    def setup_hooks():
        def _handler(exc_type, exc_value, exc_tb):
            if issubclass(exc_type, KeyboardInterrupt):
                print("\n[TERMINATED] Interrupted by user.")
                sys.exit(0)

            tb_last = exc_tb
            while tb_last.tb_next:
                tb_last = tb_last.tb_next

            print("\n" + "!" * 60)
            print(f"  ERROR : {exc_type.__name__}: {exc_value}")
            print(f"  FILE  : {os.path.basename(tb_last.tb_frame.f_code.co_filename)}")
            print(f"  FUNC  : {tb_last.tb_frame.f_code.co_name}()")
            print(f"  LINE  : {tb_last.tb_lineno}")
            print("!" * 60)
            sys.exit(1)

        sys.excepthook = _handler