const untracker = [ undefined, null ]

const Assigner = function(delegate: (a: any, b: any) => any, useuntrack: boolean = true): (...args: any[]) => any {
  const assigner = (...args: any[]): any => {
    if (!args || args.length === 0) {
      return {}
    } else {
      let a = args[0]
      let b = useuntrack ? untracker : args[1]
      if (typeof a !== 'object' || typeof b !== 'object') {
        throw new TypeError('Arguments must be objects')
      }
      const keysA = Object.keys(a)
      const keysB = Object.keys(b)
      const intersectionKeys = keysA.filter((k) => keysB.includes(k))
      const unionKeys = [...new Set([...keysA, ...keysB])]
      return unionKeys.reduce((result: any, key: string) => {
        result[key] = { ...a[key], ...b[key] }
        return result
      }, {})
    }
  }
}

const AssignerCount = function(delegate: (a: any, b: any) => any): (...args: any[]) => any {
  const assignerCount = (...args: any[]): any => {
    if (!args || args.length === 0) {
      return {}
    } else {
      let a = args[0]
      if (typeof a !== 'object') {
        throw new TypeError('Arguments must be objects')
      }
      const keysA = Object.keys(a)
      const intersectionKeys = keysA.filter((k) => keysA.includes(k))
      const unionKeys = [...new Set([...keysA])]
      return unionKeys.reduce((result: any, key: string) => {
        result[key] = 1
        return result
      }, {})
    }
  }
}
