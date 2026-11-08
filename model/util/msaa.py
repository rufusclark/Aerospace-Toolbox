"""Support and tools for Microsoft Activei Accessibility (MSAA)"""


def get_obj(handle):
    import ctypes
    import comtypes
    import comtypes.client

    from comtypes import GUID, HRESULT, COMMETHOD
    from ctypes import POINTER, c_long, c_int, c_wchar_p

    # IAccessible interface
    oleacc = ctypes.OleDLL("oleacc.dll")

    # get accessibility object from HWND (handle)
    accessible = ctypes.POINTER(comtypes.IUnknown)()

    result = oleacc.AccessibleObjectFromWindow(
        ctypes.c_void_p(handle),
        0xFFFFFFFC,  # OBJID_CLIENT
        ctypes.byref(comtypes.GUID(
            "{618736E0-3C3D-11CF-810C-00AA00389B71}"
        )),
        ctypes.byref(accessible)
    )

    if result != 0:
        raise ValueError(f"MSAA Interface Error: result == {result}")

    accessible_obj = comtypes.client.GetBestInterface(accessible)

    if len(dir(accessible_obj)) == 0:
        raise ValueError(
            f"MSAA Interface Error: No methods (type = {type(accessible_obj)})")

    return accessible_obj


def inspect_accessible(obj, depth=0, max_depth=5):
    indent = "  " * depth

    try:
        count = obj.accChildCount
        print(f"{indent}Children: {count}")

        # Inspect the object itself
        try:
            print(f"{indent}Name: {obj.accName(0)}")
            print(f"{indent}Role: {obj.accRole(0)}")
            print(f"{indent}State: {obj.accState(0)}")
        except Exception:
            pass

        if depth >= max_depth:
            return

        for i in range(1, count + 1):
            try:
                child = obj.accChild(i)

                print(f"\n{indent}Child {i}:")

                if child:
                    inspect_accessible(
                        child,
                        depth + 1,
                        max_depth
                    )
                else:
                    print(f"{indent}Name: {obj.accName(i)}")
                    print(f"{indent}Role: {obj.accRole(i)}")
                    print(f"{indent}State: {obj.accState(i)}")

            except Exception as e:
                print(f"{indent}Error on child {i}: {e}")

    except Exception as e:
        print(f"{indent}Accessibility error: {e}")
