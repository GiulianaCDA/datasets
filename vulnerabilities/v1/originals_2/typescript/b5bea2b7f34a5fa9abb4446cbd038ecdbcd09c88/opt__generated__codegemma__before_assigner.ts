const untracker = [ undefined, null ]

const Assigner = function( delegate: ( a: any, b: any ) => any, useuntrack: boolean = true ): ( ...args: any[] ) => any {
  const assigner = ( ...args: any[] ) => {
    if ( args.some( arg => untracker.includes( arg ) ) ) throw new TypeError( `can't convert any argument to object` )
    return args.reduce( ( a, b ) => {
      Object.keys( b ).forEach( key => {
        if ( useuntrack && untracker.includes( b[key] ) ) return
        a[key] = delegate.call( this, a[key], b[key] )
      } )
      return a
    } )
  }
  return assigner
}

Assigner.count = ( qty: number, delegate: ( arg: any, ...args: any[] ) => any ) => {
  const assigner = ( ...receives: any[] ) => {
    if ( receives.some( arg => untracker.includes( arg ) ) ) throw new TypeError( `can't convert any argument to object` )
    
    let group = receives.shift()
    let args = receives.splice( 0, qty - 1 )

    while ( args.length ) {
      const keys = new Set()
      for ( const arg of args )
        for ( const key of Object.keys( arg ) )
          keys.add( key )

      for ( const key of keys )
        group[key] = delegate.call( this, group[key], ...args.map( arg => arg[key] ) )

      args = receives.splice( 0, qty - 1 )
    }

    return group
  }
  return assigner
}

declare namespace Assigner {
  export type BasicTypes = string | number | symbol | bigint | object | boolean | Function
  export type Types = BasicTypes | Types[]
  export type TypesWithExclude = BasicTypes | undefined | null | TypesWithExclude[]
}

export { Assigner }
