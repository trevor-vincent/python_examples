def binary_search(arr, x):
    l = 0
    r = len(arr)
    
    while l <= r:
    
        mid = int(l + (r-l)/2)

        if arr[mid] == x:
            return mid

        elif arr[mid] < x:
            l = mid + 1

        else:
            r = mid - 1

    return None    

        

if __name__ == "__main__":
    binary_search([1,3,5,7,10],3)
